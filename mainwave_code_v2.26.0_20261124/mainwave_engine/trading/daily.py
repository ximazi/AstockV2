"""Incremental closure step after lifecycle analysis; never connects to a broker."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from . import (VERSION, store, knowledge, ranking, paper, preflight,
               continuity, diagnostics, explain, validation, rules, adds, runner,
               migrations, performance, experiments, corporate_actions, brief, books, jev, ranking_shadow, market_policy, shadow_inputs)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(store.encoded(payload), encoding="utf-8")
    temp.replace(path)


def benchmark(data_root, day):
    path = data_root / "000300_SH_index.csv"
    if not path.exists():
        return {}
    f = pd.read_csv(path)
    f["day"] = pd.to_datetime(f.trade_date.astype(str)).dt.strftime("%Y-%m-%d")
    f = f[f.day.le(day)].sort_values("day").drop_duplicates("day")
    if len(f) < 21 or f.day.iloc[-1] != day:
        return {}
    c = pd.to_numeric(f.close, errors="coerce")
    return {"name": "CSI300", "as_of": day, "return5": ranking.number((c.iloc[-1] / c.iloc[-6] - 1) * 100),
            "return20": ranking.number((c.iloc[-1] / c.iloc[-21] - 1) * 100)}


def load_features(records, data_root, day):
    metrics = {}
    for i, row in enumerate(records, 1):
        try:
            metrics[row["symbol"]] = ranking.features(ranking.prices(data_root / f"{row['symbol']}_daily.csv", day), day)
        except (OSError, ValueError, KeyError, AttributeError):
            metrics[row["symbol"]] = {"status": "DATA_WAIT"}
        if i % 200 == 0:
            print(f"[trading] rank features {i}/{len(records)}", flush=True)
    return metrics


def monitor_paper(day, db=store.DB_PATH, data_root=None):
    """EOD decisions use actual paper fills; orders run at a later session's open."""
    from .. import route_evidence as RE, exit_policy as XP
    data_root = Path(data_root or store.ROOT / "data")
    with paper.database(db) as conn:
        account = conn.execute("SELECT * FROM paper_account").fetchone()
        if not account:
            return {"status": "UNCONFIGURED", "live_orders_enabled": False}
        last = conn.execute("SELECT MAX(as_of) FROM paper_valuations").fetchone()[0]
        if last and day < last:
            return {"status": "HISTORICAL_NO_ACCOUNT_REWIND", "last_day": last}
        plans = [json.loads(r[0]) for r in conn.execute("SELECT p.payload FROM paper_plans p WHERE EXISTS (SELECT 1 FROM paper_lots l WHERE l.plan_id=p.id AND l.remaining>0) OR EXISTS (SELECT 1 FROM paper_orders o WHERE o.plan_id=p.id AND o.status IN ('OPEN','PARTIAL','EXIT_BLOCKED','UNKNOWN'))")]
        frames, bars, issues = {}, {}, []
        index_path = data_root / "000300_SH_index.csv"
        previous_market_day, index_days = None, []
        if index_path.exists():
            index = pd.read_csv(index_path)
            index_days = sorted(set(pd.to_datetime(index.trade_date.astype(str)).dt.strftime("%Y-%m-%d")))
            earlier = [d for d in index_days if d < day]
            previous_market_day = earlier[-1] if earlier and day in index_days else None
        for plan in plans:
            symbol = plan["symbol"]
            if symbol in frames:
                continue
            try:
                raw = ranking.prices(data_root / f"{symbol}_daily.csv", day)
                if raw.empty or raw.date.iloc[-1] != day:
                    raise ValueError("当日行情缺失")
                r = raw.iloc[-1]
                if not all(ranking.number(r.get(k)) and r[k] > 0 for k in ("open", "high", "low", "close", "preclose")) or r.get("volume", 0) <= 0:
                    raise ValueError("行情价格或成交量无效")
                raw["date"] = pd.to_datetime(raw.date)
                frames[symbol] = RE.build(raw, symbol)
                since = conn.execute("SELECT MIN(entry_day) FROM paper_lots WHERE symbol=? AND remaining>0", (symbol,)).fetchone()[0]
                changed = raw.preclose.div(raw.close.shift()).sub(1).abs().gt(.001)
                if since:
                    # Inferred from price only; the ledger moves after a human confirms the event kind,
                    # and only an unconfirmed event keeps blocking the position.
                    for inferred in corporate_actions.detect(raw, since=since):
                        corporate_actions.record(conn, plan["id"], inferred)
                    corporate_action = bool(corporate_actions.pending_for(conn, plan["id"]))
                else:
                    corporate_action = bool(changed.iloc[-1])
                bars[symbol] = {"date": day, "open": float(r.open), "preclose": float(r.preclose), "raw_verified": True,
                    "previous_market_day": previous_market_day,
                    "ordinary_mainboard": ranking.number(r.get("isST")) == 0 and ranking.number(r.get("tradestatus")) == 1 and bool(paper.re.fullmatch(r"(?:sh_60|sz_00)\d{4}", symbol)) and
                        not any(x in plan["name"].upper() for x in ("ST", "退")),
                    "corporate_action": corporate_action}
            except (OSError, ValueError, KeyError, AttributeError) as exc:
                issues.append(f"{symbol}: {exc}")
        # A per-day checkpoint makes reruns idempotent, including partially filled exits.
        checkpoint = "PAPER_OPEN:" + day
        if not conn.execute("SELECT 1 FROM paper_events WHERE id=?", (checkpoint,)).fetchone():
            paper.process_open(conn, day, bars)
            if not issues:
                paper.event(conn, "OPEN_PROCESSED", {"day": day}, checkpoint)
        market = XP.market_frame(index, day) if index_path.exists() else pd.DataFrame(
            columns=["market_strong", "market_weak", "market_ret20"], index=pd.DatetimeIndex([]))
        for plan in plans:
            pid = plan["id"]
            lots = list(conn.execute("SELECT * FROM paper_lots WHERE plan_id=? ORDER BY entry_day,id", (pid,)))
            remaining = sum(r["remaining"] for r in lots)
            if not lots or not remaining:
                continue
            f = frames.get(plan["symbol"])
            problems = []
            if f is None or f.empty:
                problems.append("PRICE_DATA_WAIT")
            # Changed exit code never re-judges an old plan; only its deterministic floors keep running.
            rules_review = plan["exit_code_hashes"] != paper.source_fingerprint()
            if rules_review:
                issues.append(f"{plan['symbol']}: FROZEN_RULE_MISMATCH_REQUIRES_REVIEW")
            if bars.get(plan["symbol"], {}).get("corporate_action"):
                problems.append("CORPORATE_ACTION_RECONCILIATION_REQUIRED")
            if not index_days or day not in index_days:
                problems.append("MARKET_CALENDAR_WAIT")
            if f is not None and not f.empty:
                available_days = set(f.date.dt.strftime("%Y-%m-%d"))
                if any(d not in available_days for d in index_days if lots[0]["entry_day"] <= d <= day):
                    problems.append("POSITION_HISTORY_GAP")
            if problems:
                valuation = {"day": day, "status": "DATA_WAIT", "remaining": remaining,
                             "reasons": problems + (["FROZEN_RULE_MISMATCH_REQUIRES_REVIEW"] if rules_review else [])}
                paper.event(conn, "POSITION_DATA_WAIT", {"plan_id": pid, **valuation})
                issues.extend(f"{plan['symbol']}: {x}" for x in problems)
            else:
                f = XP.enrich(f.reset_index(drop=True), market)
                days = f.date.dt.strftime("%Y-%m-%d").tolist()
                entry_day = lots[0]["entry_day"]
                if entry_day not in days:
                    raise ValueError(f"{pid}: 首仓行情缺失，不能重建退出依据")
                entry = days.index(entry_day)
                scale = float(f.close.iloc[entry] / f.index_close.iloc[entry])
                atr = plan["atr_cents"] / 100 / scale
                structure = plan["structure_cents"] / 100 / scale
                # Boundaries stay in the units of the entry day; after a confirmed share event every lot is in
                # current units, so lots map into the adjusted index space through the event factor.
                adjust = 1.
                for action in corporate_actions.applied(conn):
                    if action["plan_id"] == pid and action["applied"]["kind"] == "SHARE_EVENT":
                        adjust *= float(action["applied"]["factor"])
                lot_scale = scale / adjust

                def cost_on(d):
                    """Average cost of the lots filled up to day d, so history is replayed as it was."""
                    held = [r for r in lots if r["entry_day"] <= d]
                    return sum(r["quantity"] * r["price_cents"] for r in held) / sum(r["quantity"] for r in held) / 100 / lot_scale

                # The hard floor is frozen on the first fill; a cheaper add lowers the average, never the floor.
                price = cost_on(entry_day)
                hard = max(price - 2 * atr, plan["stop_cents"] / 100 / scale)
                a = {c: f[c].to_numpy() for c in f.columns}
                peak, soft, pending, breaches, required = price, hard, "", 0, 1
                sold_reasons = {json.loads(r[0])["reason"] for r in conn.execute("SELECT payload FROM paper_orders WHERE plan_id=? AND side='SELL' AND filled>0", (pid,))}
                reduced = "RISK_REDUCTION" in sold_reasons
                reduction = False
                for j in range(entry, len(f)):
                    price = cost_on(days[j])
                    if a["index_low"][j] <= hard:
                        pending = pending or "INITIAL_STOP"
                    if a["index_close"][j] < structure:
                        pending = pending or "INITIAL_STRUCTURE_FAILED"
                    if not rules_review:
                        review = XP.close_review(a, j, entry, price, atr, peak, soft_stop=soft, breaches=breaches,
                            required=required, pending_exit=pending, pending_reduce=reduction, reduced=reduced)
                        soft, breaches, required = (review[k] for k in ("soft_stop", "breaches", "required"))
                        pending, reduction = review["pending_exit"], review["pending_reduce"]
                    peak = max(peak, float(a["index_high"][j]))
                    if j - entry + 1 >= 60:
                        pending = pending or "HORIZON"
                reduction_reason = ""
                if pending:
                    paper.queue_exit(conn, pid, day, pending)
                elif not rules_review:
                    # Discretionary reductions and targets belong to the exit policy; they wait for review.
                    if reduction:
                        reduction_reason = "RISK_REDUCTION"
                        paper.queue_exit(conn, pid, day, "RISK_REDUCTION", .5)
                    elif peak >= price + 3 * atr and "TARGET_3ATR" not in sold_reasons:
                        reduction_reason = "TARGET_3ATR"
                        paper.queue_exit(conn, pid, day, "TARGET_3ATR", .5)
                    elif peak >= price + 6 * atr and "TARGET_6ATR" not in sold_reasons:
                        reduction_reason = "TARGET_6ATR"
                        paper.queue_exit(conn, pid, day, "TARGET_6ATR", .5)
                mark = paper.cents(f.close.iloc[-1])
                ratio = float(f.close.iloc[-1] / f.index_close.iloc[-1])
                valuation = {"day": day,
                             "status": ("EXIT_REQUIRED" if pending else "REDUCE_REQUIRED" if reduction_reason
                                        else "RULES_REVIEW_HOLD" if rules_review else "HOLD"),
                             "reason": pending or reduction_reason or ("FROZEN_RULE_MISMATCH_REQUIRES_REVIEW" if rules_review else ""),
                             "remaining": remaining, "mark_cents": mark, "market_value_cents": mark * remaining,
                             "hard_floor": hard * ratio, "soft_floor": soft * ratio,
                             "cost_basis_cents": int(round(price * lot_scale * 100)),
                             "deterministic_only": rules_review,
                             "policy_version": plan["exit_version"], "held_sessions": len(f) - entry}
            vid = store.digest([pid, valuation])
            conn.execute("INSERT OR IGNORE INTO paper_valuations VALUES (?,?,?,?)", (vid, day, pid, store.encoded(valuation)))
        # D07: adds are evaluated only after an actual fill, under the plan's dispatched rules.
        account, cfg = paper._account(conn)
        exposures = paper._exposure(conn)
        add_rows = []
        for plan in plans:
            dispatch = rules.dispatch(plan, db, conn=conn)
            frame = frames.get(plan["symbol"])
            if frame is None or frame.empty:
                add_rows.append({"plan_id": plan["id"], "status": "DATA_WAIT",
                                 "stop_reason": "PRICE_DATA_WAIT", "dispatch": dispatch["status"]})
                continue
            enriched = XP.enrich(frame.reset_index(drop=True), market)
            evaluation = adds.evaluate(conn, plan, enriched, index_days, day, exposures,
                                       account, cfg, dispatch["status"])
            evaluation["dispatch"] = dispatch["status"]
            if evaluation["status"] == "ADD_CONFIRMED":
                evaluation["placed"] = adds.queue(conn, plan, evaluation, account, cfg)
            add_rows.append(evaluation)
        return {"status": "DATA_WAIT" if issues else "SUCCESS", "issues": issues,
                "adds": add_rows, "live_orders_enabled": False}


def live_run(day, clock=None):
    """Books place orders only from `day`'s official close until 09:00 of the next weekday. A closure that runs
    past midnight is still that day's closure (its orders target the next open); a later rerun only reconstructs.
    A holiday on the next weekday closes the window early, which errs toward placing nothing."""
    clock = (clock or store.now())[:16]
    nxt = pd.Timestamp(day) + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return f"{day}T15:00" <= clock < f"{nxt.date().isoformat()}T09:00"


def evaluate_rankings(day, db=store.DB_PATH, data_root=None):
    """Delegates to the D05 evaluator; the window is fixed on market days, never slid."""
    from . import validation
    return validation.evaluate(day, db, data_root)


def run(day, db=store.DB_PATH, root=store.ROOT, monitor=True):
    day = pd.Timestamp(day).date().isoformat()
    if day > store.now()[:10] or (day == store.now()[:10] and store.now()[11:16] < "15:00"):
        raise ValueError("收盘闭环不能使用未来或未收盘日期")
    lock = runner.acquire(f"after_close:{day}", db)
    if not lock["acquired"]:
        raise ValueError(f"另一次收盘运行正在进行：{lock['owner']}（{lock.get('reason', '')}）")
    # Whatever fails, the failure is on record and the lock is released; a rerun never waits out the timeout.
    try:
        return _closure(day, db, root, monitor, lock)
    except BaseException as exc:
        try:
            runner.checkpoint(day, "AFTER_CLOSE", "closure", "FAILED", {"error": str(exc)[:500]}, db)
        except Exception:
            pass
        raise
    finally:
        runner.release(f"after_close:{day}", db)


def _closure(day, db, root, monitor, lock):
    output = root / "results/mainwave_engine" / day.replace("-", "")
    data_root = root / "data"
    # D01: required inputs are checked before anything is recorded; degraded items stay named.
    schema = migrations.upgrade(db)
    startup = preflight.check(day, root, db, data_root)
    write(output / "closure_preflight.json", startup)
    runner.checkpoint(day, "AFTER_CLOSE", "preflight", "DONE", {"status": startup["status"]}, db)
    if startup["status"] == "BLOCKED":
        raise ValueError(f"启动自检未通过，缺少必要输入：{'、'.join(startup['blocking'])}")
    universe, evidence = (read(output / name) for name in ("analysis_universe.json", "analysis_evidence.json"))
    if universe["as_of"] != day:
        raise ValueError("日报日期不匹配")
    registry = knowledge.build(root)
    registry_id = knowledge.persist(registry, db)
    rule_set = rules.register(db, root)
    with store.connect(db) as conn:
        prior = store.latest(conn, (pd.Timestamp(day) - pd.Timedelta(days=1)).date().isoformat())
    leaders = {}
    path = output / "leader_state_daily.csv"
    if path.exists():
        f = pd.read_csv(path).fillna("")
        for row in f.to_dict("records"):
            if row.get("as_of") == day:
                leaders.setdefault(row.get("leader"), []).append(row)
    payload = ranking.build(universe, evidence, load_features(universe["records"], data_root, day), benchmark(data_root, day), prior, leaders)
    if day == store.now()[:10]:
        payload["observation_basis"] = "FORWARD_RECORDED_AFTER_CLOSE"
    payload["knowledge_registry_id"] = registry_id
    payload["code_sha256"] = hashlib.sha256(Path(ranking.__file__).read_bytes()).hexdigest()
    payload["source_hashes"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("analysis_universe.json", "analysis_evidence.json")}
    snap = store.persist(payload, db)
    write(root / "results/trading_system/knowledge_registry.json", registry)
    write(output / "trading_closure.json", snap)
    sessions = preflight._calendar(data_root, day)[0] or []
    # D02-D05: continuity, per-step route cards, rank explanation and forward validation.
    chain = continuity.advance(snap, db, sessions)
    write(output / "opportunity_continuity.json", chain)
    # Shadow layer: layered screening, market policy dimensions, entry-contract drafts, discovery register.
    # None of it changes states, ranks or permissions; a failure here never blocks the closure.
    shadow, watch = None, []
    try:
        prior_day = str((prior or {}).get("as_of", "")).replace("-", "")
        prior_shadow_path = root / "results/mainwave_engine" / prior_day / "ranking_shadow.json"
        watch_path = output / "preparation_watch.csv"
        watch = pd.read_csv(watch_path, encoding="utf-8-sig").to_dict("records") if watch_path.exists() else []
        prior_facts = shadow_inputs.previous_review_facts(root, prior_day)
        shadow = ranking_shadow.evaluate(snap, read(prior_shadow_path) if prior and prior_shadow_path.exists() else None, watch,
                                         theme_facts=prior_facts.get("themes"))
        write(output / "ranking_shadow.json", shadow)
        policy = market_policy.evaluate(snap.get("market", {}), (prior_facts.get("sentiment") or {}).get("metrics"),
                                        (prior_facts.get("loss_effects") or {}).get("events"), prior_facts.get("as_of"))
        write(output / "market_policy.json", policy)
        drafts = shadow_inputs.contract_drafts(snap, day, sessions, output)
        write(output / "entry_contract_drafts.json", drafts)
        shadow_summary = {"status": "SUCCESS", **{k: shadow["summary"][k] for k in ("total", "re_eligible", "cohort_strong", "top10_overlap_with_frozen")},
                          "review_shortlist": [x["symbol"] for x in shadow["summary"]["review_shortlist"]],
                          "preparation_watch": shadow["summary"]["preparation_watch"], "contract_drafts": drafts["summary"],
                          "market_policy": {e: policy["policy"].get(e, {}).get("shadow") for e in ("E1", "E2", "E3")}}
    except Exception as exc:  # a shadow must never block the closure
        shadow_summary = {"status": "FAILED", "error": str(exc)[:300]}
    cards = diagnostics.report(snap)
    write(output / "route_diagnostics.json", cards)
    board = explain.board(snap, prior)
    write(output / "ranking_explained.json", board)
    runner.renew(f"after_close:{day}", db)
    review = validation.evaluate(day, db, data_root)
    write(output / "ranking_validation.json", review)
    audit = {"protocols": validation.protocols(root), "ablation": validation.ablate(day, root),
             "missed": validation.missed_audit(day, db, data_root)}
    write(output / "validation_audit.json", audit)
    runner.renew(f"after_close:{day}", db)
    paper_status = monitor_paper(day, db, data_root) if monitor else {"status": "NOT_RUN"}
    # Three books apart: SIGNAL records the mechanical trade set, AUTO executes it without a person,
    # MANUAL is this ledger. Only a same-day run lets the auto book place orders.
    signal = books.record_signal(day, snap, db)
    auto = (books.run_auto(day, payload, snap, root, data_root, live=live_run(day), signal=signal, main_db=db)
            if monitor else {"status": "NOT_RUN", "book": "AUTO"})
    try:
        new100k = (books.run_new100k(day, payload, snap, root, data_root, live=live_run(day), signal=signal, main_db=db)
                   if monitor else {"status": "NOT_RUN", "book": "NEW100K"})
    except Exception as exc:  # the compounding book must never block the closure or the other books
        new100k = {"status": "FAILED", "book": "NEW100K", "error": str(exc)[:300]}
    try:
        register = shadow_inputs.discovery_register(root, day, snap, shadow, watch, auto, sessions)
        write(output / "discovery_events.json", register)
        shadow_summary["discovery"] = register["summary"]
    except Exception as exc:  # the register is an observation artefact; its failure is recorded, not raised
        shadow_summary["discovery"] = {"status": "FAILED", "error": str(exc)[:300]}
    status = {"as_of": day, "version": VERSION, "snapshot_id": snap["snapshot_id"], "ranking": snap["summary"],
              "validation": review["summary"], "knowledge": registry["summary"], "paper": paper_status,
              "preflight": {"status": startup["status"], "blocking": startup["blocking"],
                            "degraded": startup["degraded"]},
              "continuity": chain["summary"], "diagnostics": cards["summary"], "shadow_screening": shadow_summary,
              "explained": board["summary"],
              "audit": {"ablation": audit["ablation"]["status"], "missed": audit["missed"]["status"]},
              "rules": {"fingerprint": rule_set["fingerprint"], "software": rule_set["software_version"]},
              "live_orders_enabled": False, "status": "PARTIAL" if paper_status["status"] == "DATA_WAIT" else "SUCCESS"}
    perf = performance.report(day, db)
    write(output / "paper_performance.json", perf)
    manual = {"book": "MANUAL", "status": paper_status["status"], "performance": perf["status"],
              **({k: perf["account"][k] for k in ("equity_cents", "net_return", "max_drawdown", "marked")}
                 if perf.get("account") else {})}
    write(output / "books_daily.json", {"version": books.VERSION, "as_of": day, "books": books.BOOKS,
                                        "signal": signal, "auto": auto, "manual": manual, "new100k": new100k,
                                        "separation": "三本账各自回答一个问题，成绩不合并。"})
    status["books"] = {"signal": {"trade_set": [r["symbol"] for r in signal["trade_set"]], "eligible": signal["eligible"],
                                  "excluded_r3_research": signal["excluded_r3_research"],
                                  "snapshot_id": signal["snapshot_id"], "rerun": signal.get("rerun")},
                       "auto": {k: auto.get(k) for k in ("live", "ledger_touched", "plans", "open_positions", "fills",
                                                         "reconciliation", "account_matches_code", "snapshot_id")}
                               | {"placed": [(p["symbol"], p["status"]) for p in auto.get("placed", [])]},
                       "manual": manual}
    reviewed = {p.parent.name[:4] + "-" + p.parent.name[4:6] + "-" + p.parent.name[6:]
                for p in (root / "results/mainwave_engine").glob("*_v1_*/reviews/*/interpretation.md")}
    status["performance"] = {"status": perf["status"], **({k: perf["account"][k] for k in ("equity_cents", "net_return", "max_drawdown", "marked")}
                                                         if perf.get("account") else {})}
    status["ops"] = {"schema_version": schema["to"], "corporate_actions_pending": len(corporate_actions.pending(db)),
                     "review_cadence": experiments.cadence(day, done=sorted(reviewed))}
    status["run"] = {"lock": lock, "freshness": runner.freshness(day, data_root, output)["status"],
                     "reconciliation": runner.reconcile(day, db)["status"],
                     "pending_tasks": len(runner.pending(day, db)["tasks"])}
    # Jev shadow: semantic judgments recorded beside the ranking; nothing downstream reads them yet.
    jev_shadow = jev.run_shadow(snap, db) if monitor else {"status": "NOT_RUN", "rows": [], "summary": {}}
    jev_review = jev.evaluate(day, db, data_root)
    write(output / "jev_shadow.json", jev_shadow)
    write(output / "jev_validation.json", jev_review)
    (output / "jev_shadow.md").write_text(jev.render(jev_shadow, jev_review), encoding="utf-8")
    status["jev"] = {"status": jev_shadow["status"], **{k: jev_shadow.get("summary", {}).get(k) for k in
                                                        ("recorded", "errors", "would_review", "would_wait_late", "cost_usd")},
                     "validation": jev_review["status"], "mature_rows": jev_review["mature_rows"]}
    # The one deliverable: Top3/5 + per-stock steps + cross-day change + outcome comparison, stamped.
    deliverable = brief.build(day, root)
    write(output / "daily_brief.json", deliverable)
    (output / "daily_brief.md").write_text(brief.render(deliverable), encoding="utf-8")
    status["brief"] = {"screening_ability_proven": deliverable["screening_ability_proven"],
                       "missing_inputs": deliverable["missing_inputs"],
                       "stamps": {k: {j: v["stamp"][j] for j in ("software", "historical", "forward")}
                                  for k, v in deliverable["sections"].items()}}
    runner.checkpoint(day, "AFTER_CLOSE", "closure", "DONE", {"snapshot_id": snap["snapshot_id"]}, db)
    write(output / "run_status.json", runner.status(day, db, data_root))
    write(output / "trading_closure_status.json", status)
    compact = [{k: r.get(k) for k in ("strength_rank", "participation_rank", "symbol", "name", "theme", "route_ids", "role", "state", "strength_score", "rank_change")}
               for r in sorted(snap["records"], key=lambda r: r["participation_rank"])[:15]]
    text = [f"# 机会与模拟交易闭环 {day}", "", "真实下单关闭；排名基线未校准，不代表收益概率。", "",
            f"知识登记：{registry['summary']}；候选：{snap['summary']}。", f"人工账：{paper_status['status']}。",
            f"信号账交易集：{[r['symbol'] for r in signal['trade_set']] or '空集'}；自动账：{auto.get('placed', []) and [(p['symbol'], p['status']) for p in auto['placed']] or '无下单'}。",
            "", pd.DataFrame(compact).to_markdown(index=False) if compact else "无候选", "",
            f"排序验证：{review['summary']}。", review["notice"], "", "平台入口：http://127.0.0.1:8068/trading"]
    (output / "trading_closure.md").write_text("\n".join(text), encoding="utf-8")
    print(f"[trading] {day} candidates={len(snap['records'])} eligible={snap['summary']['paper_plan_eligible']} paper={paper_status['status']} validation={review['summary']}", flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--db", type=Path, default=store.DB_PATH)
    parser.add_argument("--no-paper", action="store_true", help="只回建研究快照，不重放模拟账户")
    args = parser.parse_args()
    result = run(args.date, args.db, monitor=not args.no_paper)
    raise SystemExit(0 if result["status"] == "SUCCESS" else 1)
