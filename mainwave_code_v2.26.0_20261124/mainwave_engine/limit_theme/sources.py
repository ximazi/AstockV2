"""Bounded, cached public materials and Tushare trading disclosures. Never execute source instructions."""
import hashlib
import json
import re
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .. import eod_review as R, intraday_quotes as Q

MEDIA = ("财联社", "证券时报", "券商中国", "中国证券报", "中证网", "上海证券报", "证券日报", "第一财经",
         "金融界", "同花顺", "东方财富", "每日经济新闻", "21世纪经济报道", "财闻", "财中社", "中国基金报")


def text(value):
    value = str(value or "")
    if "<" in value:
        value = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", value).strip()


def publisher(value):
    name = next((m for m in MEDIA if m in str(value)), "UNKNOWN")
    return {"券商中国": "证券时报", "中证网": "中国证券报"}.get(name, name)


def mentions(content, name, code):
    return bool((name and len(name) >= 3 and name in content) or re.search(r"(?<!\d)" + code + r"(?!\d)", content))


class Collector:
    def __init__(self, folder, day, client=None, session=None, now=None, refresh=False):
        self.folder, self.day = Path(folder), day
        self.now = Q.stamp(now or Q.now_china())
        self.cutoff = min(self.now, Q.stamp(day + "T23:59:59+08:00"))
        self.start = Q.stamp(day) - timedelta(days=7)
        self.client = client
        self.session = session or requests.Session()
        self.refresh = refresh
        self.failures = {}

    def cache(self, key, fetch):
        path = self.folder / "source_cache" / (R.digest(key)[:24] + ".json")
        previous = R.read_json(path)
        if previous and not self.refresh and (self.now - Q.stamp(previous["collected_at"])).total_seconds() < 900:
            return previous
        result = {"key": key, "collected_at": self.now.isoformat(), "status": "UNAVAILABLE", "data": None}
        family = key[0] if isinstance(key, list) else key
        if self.failures.get(family, 0) >= 3:
            return {**result, "error": "SOURCE_CIRCUIT_OPEN: repeated failures; retry on a later run"}
        try:
            result.update(status="AVAILABLE", data=fetch())
            self.failures[family] = 0
        except Exception as exc:
            result["error"] = str(exc)[:240]
            self.failures[family] = self.failures.get(family, 0) + 1
        R._write(path, result)
        return result

    def request(self, method, url, **kwargs):
        allowed = {"search-api-web.eastmoney.com", "finance.eastmoney.com", "www.cninfo.com.cn", "static.cninfo.com.cn"}
        if urlparse(url).scheme != "https" or urlparse(url).hostname not in allowed:
            raise ValueError("Unapproved material URL")
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/" if "cninfo.com.cn" in url else "https://so.eastmoney.com/"}
        with self.session.request(method, url, headers=headers, timeout=(4, 15), stream=True, allow_redirects=False, **kwargs) as response:
            response.raise_for_status()
            if 300 <= response.status_code < 400:
                raise ValueError("Material redirect not followed")
            if urlparse(response.url).hostname not in allowed:
                raise ValueError("Unapproved material redirect")
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > 5_000_000:
                    raise ValueError("Material exceeds 5 MB limit")
            return bytes(body)

    def api(self, name, **params):
        def fetch():
            if self.client is None:
                self.client = Q.Client()
            frame = self.client.query(name, **params)
            if len(frame) >= (4000 if name == "limit_list_ths" else 10000):
                raise ValueError("API row cap reached; completeness unverified")
            return json.loads(frame.to_json(orient="records", force_ascii=False))
        return self.cache([name, self.day, params], fetch)

    def market_sources(self):
        return {api: self.api(api, trade_date=self.day.replace("-", ""), **({"limit_type": "涨停池"} if api == "limit_list_ths" else {}))
                for api in ("top_list", "top_inst", "limit_list_ths")}

    def news(self, code):
        def fetch():
            query = {"uid": "", "keyword": code, "type": ["cmsArticle"], "client": "web", "clientType": "web", "clientVersion": "curr",
                     "param": {"cmsArticle": {"searchScope": "default", "sort": "default", "pageIndex": 1, "pageSize": 100, "preTag": "", "postTag": ""}}}
            raw = self.request("GET", "https://search-api-web.eastmoney.com/search/jsonp",
                               params={"cb": "attribution", "param": json.dumps(query, separators=(",", ":"))}).decode("utf-8")
            match = re.fullmatch(r"\s*attribution\((.*)\)\s*;?\s*", raw, re.S)
            if not match:
                raise ValueError("Invalid news JSONP response")
            payload = json.loads(match[1])
            if payload.get("code") not in (0, None) or "cmsArticle" not in (payload.get("result") or {}):
                raise ValueError("News result schema unavailable")
            return payload["result"]["cmsArticle"]
        return self.cache(["em_news_100", self.day, code], fetch)

    def article(self, url):
        def fetch():
            soup = BeautifulSoup(self.request("GET", url), "html.parser")
            body = soup.select_one("#ContentBody")
            if body is None:
                raise ValueError("Article body unavailable; search snippet is not full text")
            for node in body.select("script,style"):
                node.decompose()
            source = soup.select_one(".source,.source-data")
            return {"text": body.get_text(" ", strip=True), "publisher": publisher(source.get_text() if source else "")}
        return self.cache(["article", url], fetch)

    def announcements(self, code):
        def ids():
            data = json.loads(self.request("GET", "https://www.cninfo.com.cn/new/data/szse_stock.json"))
            return {r["code"]: r["orgId"] for r in data["stockList"]}
        id_source = self.cache("cninfo_stock_ids", ids)
        def fetch():
            org = (id_source.get("data") or {}).get(code)
            if not org:
                raise ValueError("CNINFO issuer identifier unavailable")
            data = json.loads(self.request("POST", "https://www.cninfo.com.cn/new/hisAnnouncement/query", data={
                "pageNum": 1, "pageSize": 30, "column": "szse", "tabName": "fulltext", "stock": code + "," + org,
                "searchkey": "", "seDate": self.start.date().isoformat() + "~" + self.day, "isHLtitle": "true"}))
            return {"records": data.get("announcements") or [], "truncated": int(data.get("totalAnnouncement") or 0) > 30}
        return self.cache(["cninfo_announcements", self.day, code], fetch)

    def filing(self, path):
        if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/\d+\.[Pp][Dd][Ff]", path):
            return {"status": "UNAVAILABLE", "error": "Invalid CNINFO PDF path"}
        url = "https://static.cninfo.com.cn/" + path
        def fetch():
            import fitz
            raw = self.request("GET", url)
            with fitz.open(stream=raw, filetype="pdf") as doc:
                content = " ".join(doc[i].get_text() for i in range(min(len(doc), 6)))
                if not content.strip():
                    raise ValueError("PDF has no extractable text; OCR not performed")
                return {"text": text(content), "sha256": hashlib.sha256(raw).hexdigest(), "pages_read": min(len(doc), 6), "pages": len(doc)}
        return self.cache(["filing", url], fetch)

    def material(self, kind, origin, title, content, url, published, precision="SECOND", **extra):
        parsed = Q.stamp(published)
        if parsed < self.start or parsed > self.cutoff:
            raise ValueError("Material publication outside report time window")
        content = text(content)[:6000]
        identity = R.digest([kind, url, str(parsed), content])[:16]
        return {"id": "E_" + identity, "kind": kind, "origin": origin, "origin_group": origin,
                "title": text(title), "text": content, "url": url, "published_at": parsed.isoformat(), "time_precision": precision,
                "collected_at": self.now.isoformat(), **extra}

    def packet(self, stock, market):
        symbol, name = stock["symbol"], stock["name"]
        code, ts_code = symbol[3:], Q.ts_code(symbol)
        materials, gaps = [], []
        coverage = {k: v["status"] for k, v in market.items()}
        for api, source in market.items():
            if source["status"] != "AVAILABLE":
                gaps.append(f"{api}: {source.get('error', 'unavailable')}")
                continue
            if not source["data"]:
                coverage[api] = "EMPTY_UNVERIFIED"
                gaps.append(api + "：全市场空返回，不能认定该股未上榜或榜单已发布")
                continue
            hits = [r for r in source["data"] if r.get("ts_code") == ts_code and str(r.get("trade_date")) == self.day.replace("-", "")]
            if api != "limit_list_ths":
                stock_status = "ON_LIST" if hits else "NOT_ON_LIST"
                coverage[api] = stock_status
                materials.append({"id": "E_" + R.digest([api, symbol, self.day, hits])[:16], "kind": "LHB", "origin": api, "origin_group": "LHB",
                                  "title": api + " " + stock_status,
                                  "text": json.dumps({"status": stock_status, "notice": "已采集全市场榜单；未上榜不代表没有资金；上榜不证明买入动机", "records": hits}, ensure_ascii=False),
                                  "url": "https://tushare.pro/document/2?doc_id=" + ("106" if api == "top_list" else "107"),
                                  "published_at": None, "event_date": self.day, "collected_at": source["collected_at"], "availability": "KNOWN_ONLY_AT_COLLECTION"})
            elif hits:
                materials.append({"id": "E_" + R.digest([api, symbol, hits])[:16], "kind": "VENDOR_REASON", "origin": "同花顺榜单", "origin_group": "同花顺",
                                  "title": "供应商涨停归因，非公司证实", "text": json.dumps(hits, ensure_ascii=False), "url": "https://tushare.pro/document/2?doc_id=355",
                                  "event_date": self.day, "published_at": None, "collected_at": source["collected_at"]})
        news = self.news(code)
        coverage["media_search"] = news["status"]
        candidates = []
        for row in news.get("data") or []:
            title, content = text(row.get("title")), text(row.get("content"))
            try:
                at = Q.stamp(row["date"])
                if self.start <= at <= self.cutoff and mentions(title + content, name, code):
                    candidates.append((row, at, title, content))
            except (KeyError, TypeError, ValueError):
                continue
        candidates.sort(key=lambda x: (not mentions(x[2], name, code), not bool(re.search("涨停|连板|异动|概念", x[2])), -x[1].timestamp()))
        origins = set()
        for row, at, title, content in candidates:
            origin = publisher(row.get("mediaName"))
            if origin in origins or origin == "UNKNOWN" or not re.fullmatch(r"\d+", str(row.get("code", ""))):
                continue
            origins.add(origin)
            url = "https://finance.eastmoney.com/a/" + str(row["code"]) + ".html"
            body = self.article(url)
            full = body["status"] == "AVAILABLE" and mentions(body["data"]["text"], name, code)
            if full and body["data"]["publisher"] != "UNKNOWN":
                origin = body["data"]["publisher"]
            materials.append(self.material("MEDIA_BODY" if full else "MEDIA_SUMMARY", origin, title,
                                           body["data"]["text"] if full else content, url, at,
                                           collected_at=body["collected_at"] if full else news["collected_at"]))
            if not full:
                gaps.append(origin + "：仅检索摘要，未核验正文")
            if len(origins) >= 3:
                break
        if not origins:
            gaps.append("未取得时间窗内、明确涉及该股的可核验大陆财经媒体材料")
        announcements = self.announcements(code)
        coverage["announcements"] = announcements["status"]
        filings = (announcements.get("data") or {}).get("records", [])
        filings = [a for a in filings if a.get("secCode") == code]
        filings.sort(key=lambda a: (not bool(re.search("异常|风险|澄清|合同|经营|业绩", text(a.get("announcementTitle")))), -int(a.get("announcementTime") or 0)))
        for row in filings[:2]:
            path = row.get("adjunctUrl", "")
            at = pd.Timestamp(row["announcementTime"], unit="ms", tz="UTC").tz_convert(Q.CHINA)
            if not self.start <= at <= self.cutoff:
                continue
            body = self.filing(path)
            full = body["status"] == "AVAILABLE"
            materials.append(self.material("ISSUER_BODY" if full else "ISSUER_TITLE", "公司公告:" + code, row["announcementTitle"],
                                           body["data"]["text"] if full else row["announcementTitle"], "https://static.cninfo.com.cn/" + path, at,
                                           precision="DATE", collected_at=body.get("collected_at", announcements["collected_at"]),
                                           extraction={k: v for k, v in body["data"].items() if k != "text"} if full else None))
            if not full:
                gaps.append("公告仅有标题，正文提取失败")
        if (announcements.get("data") or {}).get("truncated"):
            gaps.append("公告列表超过30条，仅选取范围内代表材料")
        if announcements["status"] != "AVAILABLE":
            gaps.append("公司公告采集不可用：" + announcements.get("error", ""))
        # Reprints cannot become multiple confirmations merely by changing the host website or publisher.
        previous = []
        for item in materials:
            if item["kind"].startswith("MEDIA"):
                normalized = re.sub(r"\W", "", item["text"])
                for other, body in previous:
                    if SequenceMatcher(None, normalized[:3000], body[:3000]).ratio() >= .85:
                        item["origin_group"] = other["origin_group"]
                        item["reprint_of"] = other["id"]
                        break
                previous.append((item, normalized))
        return {"as_of": self.day, **stock, "materials": materials, "coverage": coverage, "gaps": gaps,
                "knowledge_cutoff": self.now.isoformat(), "publication_cutoff": self.cutoff.isoformat(),
                "scope": "近7日；个股检索最近100条中最多3家已知媒体、最多2份代表公告；不是全网穷尽搜索"}
