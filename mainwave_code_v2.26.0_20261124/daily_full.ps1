# Unified entry: official after-close pipeline, or isolated provisional intraday review.
# .\daily_full.ps1
# .\daily_full.ps1 -Intraday   (09:30-11:30 and 13:00-16:00 China time; may be run several times a day, each run is kept)
# .\daily_full.ps1 -AuctionWatch   (launch 09:15 CST for notices and the opening-auction observations)
# .\daily_full.ps1 -Date 2026-09-11 -SkipUpdate -RebuildMainwave
# Jev defaults to http://127.0.0.1:7890; -JevProxy can override it for this run.
# Updates observations and the isolated paper ledger if configured; no live broker orders.
param(
    [string]$Date = "",
    [string]$Capital = "",
    [string]$Positions = "",
    [string]$Portfolio = "",
    [switch]$RebuildMainwave,
    [switch]$RebuildWeeklyTrace,
    [switch]$SkipUpdate,
    [switch]$Intraday,
    [switch]$IntradayWatch,
    [switch]$AuctionWatch,
    [ValidateSet("market", "pool")][string]$IntradayScope = "market",
    [switch]$ProbeRealtime,
    [string]$JevProxy = "http://127.0.0.1:7890"
)
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
if ($JevProxy) { $env:JEV_PROXY = $JevProxy }
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
Set-Location -LiteralPath $PSScriptRoot
if ($Capital -or $Positions -or $Portfolio) {
    Write-Warning "Capital/Positions/Portfolio are retired options and are ignored. Analysis only."
}
if (-not $Date) { $Date = Get-Date -Format "yyyy-MM-dd" }
$Compact = ($Date -replace '[-/]', '')
if ($Compact -notmatch '^\d{8}$') { Write-Error "Use YYYY-MM-DD or YYYYMMDD"; exit 1 }
$Dash = "{0}-{1}-{2}" -f $Compact.Substring(0,4), $Compact.Substring(4,2), $Compact.Substring(6,2)
try { $null = Get-Date -Date $Dash -ErrorAction Stop }
catch { Write-Error "Invalid date: $Date"; exit 1 }
if ($AuctionWatch) {
    if ($Intraday -or $IntradayWatch -or $ProbeRealtime -or $SkipUpdate -or $RebuildMainwave -or $RebuildWeeklyTrace) {
        Write-Error "AuctionWatch is an isolated observation mode."
        exit 1
    }
    & python -m mainwave_engine.board_auction --date $Dash --watch
    exit $LASTEXITCODE
}
if ($Intraday -or $IntradayWatch -or $ProbeRealtime) {
    if ($ProbeRealtime -and ($Intraday -or $IntradayWatch)) {
        Write-Error "Use either a realtime permission probe or an intraday run."
        exit 1
    }
    if ($SkipUpdate -or $RebuildMainwave -or $RebuildWeeklyTrace) {
        Write-Error "Intraday/probe cannot be combined with after-close rebuild/update switches."
        exit 1
    }
    $intradayArgs = @("-m", "mainwave_engine.intraday_review", "--date", $Dash)
    if ($ProbeRealtime) { $intradayArgs += "--probe" }
    else { $intradayArgs += @("--scope", $IntradayScope) }
    if ($IntradayWatch) { $intradayArgs += "--watch" }
    & python @intradayArgs
    $intradayExit = $LASTEXITCODE
    if ($intradayExit -eq 0 -and -not $ProbeRealtime) {
        # Join the provisional intraday read to the frozen closing opportunities; observation only.
        & python -m mainwave_engine.trading.intraday_link --date $Dash
        if ($LASTEXITCODE -ne 0) { Write-Warning "Intraday link failed; the after-close result is unaffected." }
    }
    exit $intradayExit
}
Write-Host "Target trading day: $Dash | lifecycle engine | analysis only" -ForegroundColor Cyan
$out = "results\mainwave_engine\$Compact"
$runStarted = Get-Date
$script:RunSteps = @()
$script:RunIssues = @()

function Write-DailyStatus([string]$State) {
    $payload = [ordered]@{
        as_of = $Dash
        version = "daily_full_v1_8_20261124"
        status = $State
        started_at = $runStarted.ToString("s")
        updated_at = (Get-Date).ToString("s")
        seconds = [math]::Round(((Get-Date) - $runStarted).TotalSeconds, 2)
        skip_update = [bool]$SkipUpdate
        steps = @($script:RunSteps)
        issues = @($script:RunIssues)
    }
    $null = New-Item -ItemType Directory -Path $out -Force -ErrorAction Stop
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$out\daily_full_status.json" -Encoding UTF8 -ErrorAction Stop
}

function Invoke-DailyPython([string]$Name, [string[]]$CommandArgs) {
    $started = Get-Date
    Write-Host "  > python $($CommandArgs -join ' ')" -ForegroundColor DarkGray
    try {
        & python @CommandArgs | Out-Host
        $code = $LASTEXITCODE
    }
    catch {
        Write-Warning "${Name}: $_"
        $code = 1
    }
    $script:RunSteps += [pscustomobject]@{
        name = $Name
        status = $(if ($code -eq 0) { "SUCCESS" } else { "FAILED" })
        exit_code = $code
        seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 2)
        command = @("python") + $CommandArgs
    }
    Write-DailyStatus "RUNNING"
    return $code
}

function Show-MarketRisk {
    # Market trend-exit alert from the position monitor (report only; never places or changes orders).
    $riskPath = "results\trading_system\position_monitor\$Compact.json"
    try { $monitor = Get-Content $riskPath -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json }
    catch { return }
    $risk = $monitor.market_risk
    $warning = $monitor.market_warning
    if ($warning -and $warning.level -eq "WARNING") {
        $hit = @($warning.indices | Where-Object { $_.level -eq "WARNING" } | ForEach-Object { "day $($_.ma20_falling_days)" }) -join ","
        Write-Host "  PULLBACK WARNING: MA20 just turned down on a broad index [$hit]; first 6 sessions weaker on average. Alert only." -ForegroundColor Black -BackgroundColor DarkYellow
    }
    $rebound = $monitor.rebound_basket
    if ($rebound -and $rebound.level -eq "TRIGGERED_TODAY") {
        $bar = "=" * 78
        Write-Host $bar -ForegroundColor White -BackgroundColor DarkBlue
        Write-Host "  REBOUND BASKET (P2) TRIGGERED -- CSI1000 $([math]::Round($rebound.drawdown_60d * 100, 1))% below its 60-day high" -ForegroundColor White -BackgroundColor DarkBlue
        Write-Host "  Buy 1/3 at tomorrow's close, hold 20 sessions, separate book. ALERT ONLY - no orders." -ForegroundColor White -BackgroundColor DarkBlue
        Write-Host "  Candidates: results\trading_system\position_monitor\$Compact.md" -ForegroundColor White -BackgroundColor DarkBlue
        Write-Host $bar -ForegroundColor White -BackgroundColor DarkBlue
    }
    elseif ($rebound -and $rebound.level -eq "ACTIVE") {
        Write-Host "Rebound basket (P2) active since $($rebound.leg_start): CSI1000 $([math]::Round($rebound.drawdown_60d * 100, 1))% below 60-day high" -ForegroundColor Cyan
    }
    $dip = $monitor.etf_dip
    if ($dip -and $dip.level -eq "TRIGGERED_TODAY") {
        $names = @($dip.new | ForEach-Object { "$($_.etf) $($_.signal)" }) -join ", "
        Write-Host "  ETF DIP SIGNAL: buy at tomorrow's open per the dip-sleeve rules [$names]; hold 20 sessions. ALERT ONLY." -ForegroundColor White -BackgroundColor DarkGreen
    }
    elseif ($dip -and $dip.level -eq "HOLDING") {
        Write-Host "ETF dip sleeve holds $(@($dip.holding).Count) position(s) per the rules (exposure $([math]::Round($dip.exposure * 100, 1))% of the sleeve)" -ForegroundColor Green
    }
    if (-not $risk) { return }
    if ($risk.level -eq "TREND_EXIT") {
        $bar = "!" * 78
        Write-Host $bar -ForegroundColor White -BackgroundColor DarkRed
        Write-Host "  MARKET RISK: TREND EXIT -- CSI1000 close $($risk.close) < MA20 $($risk.ma20), MA20 falling (5d ago $($risk.ma20_5d_ago))" -ForegroundColor White -BackgroundColor DarkRed
        Write-Host "  Rule G1: clear positions and stop new buys. ALERT ONLY - no automatic orders." -ForegroundColor White -BackgroundColor DarkRed
        Write-Host "  Details: results\trading_system\position_monitor\$Compact.md" -ForegroundColor White -BackgroundColor DarkRed
        Write-Host $bar -ForegroundColor White -BackgroundColor DarkRed
    }
    elseif ($risk.level -eq "CRITICAL") {
        Write-Host "  MARKET RISK: CRITICAL -- CSI1000 close $($risk.close) < MA20 $($risk.ma20); a flat close tomorrow turns MA20 down (G1). No new buys. ALERT ONLY." -ForegroundColor Black -BackgroundColor DarkYellow
    }
    elseif ($risk.level -eq "NORMAL") {
        Write-Host "Market risk: trend intact (CSI1000 close $($risk.close) vs MA20 $($risk.ma20))" -ForegroundColor Green
    }
    else {
        Write-Host "MARKET RISK UNCONFIRMED ($($risk.level)): CSI1000 data not current; do not treat as normal." -ForegroundColor Black -BackgroundColor Yellow
    }
}

Write-DailyStatus "RUNNING"
Write-Host "Included: data/indices/hotspots, virtual holdings, three routes, lifecycle/price-volume, daily candidates, ETFs."
Write-Host "No separate daily commands; viewer and historical research are not started by this script."

if (-not $SkipUpdate) {
    Write-Host "`n========== STEP 1: DATA UPDATE (update first, then analyze -> no write race) ==========" -ForegroundColor Magenta

    Write-Host "`n[1/6] stock daily (x_stock_data_updater)..." -ForegroundColor Cyan
    # The listing file only refreshes when it is stale AND we are running for today:
    # tushare stock_basic returns current listings, so a historical rerun must not rewrite it.
    # Without this the list never updates and new IPOs never get daily bars downloaded.
    $listArgs = @("--skip-list")
    if ($Compact -eq (Get-Date -Format "yyyyMMdd") -and (Test-Path "data\stock_list.csv")) {
        $listAgeDays = ((Get-Date) - (Get-Item "data\stock_list.csv").LastWriteTime).Days
        if ($listAgeDays -gt 7) {
            Write-Host "  stock_list.csv is $listAgeDays days old; refreshing listings (IPOs/delistings)." -ForegroundColor Yellow
            $listArgs = @()
        }
    }
    $stockExit = Invoke-DailyPython "stock_daily" (@("x_stock_data_updater.py") + $listArgs)
    if ($stockExit -ne 0) {
        $script:RunIssues += "Stock update failed; downstream analysis was not run. Existing reports are not this run's results."
        Write-DailyStatus "FAILED"
        Write-Host "========== DAILY RUN FAILED ==========" -ForegroundColor Red
        Write-Host "Date: $Dash | failed: stock_daily | downstream: NOT RUN"
        Write-Host "Health: $out\daily_full_status.json"
        exit $stockExit
    }

    Write-Host "`n[2/6] market breadth (market_daily_stats, used by regime/panic and index freshness)..." -ForegroundColor Cyan
    $code = Invoke-DailyPython "market_breadth" @("update_market_stats.py")
    if ($code -ne 0) { Write-Host "  [WARN] market breadth non-zero exit, continue." -ForegroundColor Yellow }

    Write-Host "`n[3/6] major indices..." -ForegroundColor Cyan
    $code = Invoke-DailyPython "major_indices" @("update_a_share_major_indices.py", "--end-date", $Compact)
    if ($code -eq 2) {
        Write-Host "  [WARN] some indices are still missing after retries; the lifecycle pipeline checks 000300 before scanning." -ForegroundColor Yellow
        $script:RunIssues += "Major index refresh incomplete for $Dash (see the major_indices log). Retry: python update_a_share_major_indices.py --end-date $Compact"
    }
    elseif ($code -ne 0) { Write-Host "  [WARN] major indices non-zero exit, continue." -ForegroundColor Yellow }

    Write-Host "`n[4/6] sector indices..." -ForegroundColor Cyan
    $code = Invoke-DailyPython "sector_indices" @("update_a_share_sector_indices.py", "--end-date", $Compact)
    if ($code -ne 0) { Write-Host "  [WARN] sector indices non-zero exit, continue." -ForegroundColor Yellow }

    Write-Host "`n[5/6] concept mapping freshness..." -ForegroundColor Cyan
    $mappingNeedsRefresh = $true
    if (Test-Path "concept_mapping.json") {
        try {
            $mappingPayload = Get-Content "concept_mapping.json" -Raw -Encoding UTF8 | ConvertFrom-Json
            $mappingGenerated = [datetime]::Parse([string]$mappingPayload.generated)
            # The full taxonomy is the stable monthly base. fetch-hotspot below maintains
            # a lightweight daily overlay for fast-moving themes.
            $mappingNeedsRefresh = (((Get-Date).Date - $mappingGenerated.Date).Days -gt 30)
        }
        catch {
            Write-Host "  [WARN] concept mapping metadata is invalid; refresh is recommended." -ForegroundColor Yellow
        }
    }
    $TodayCompact = Get-Date -Format "yyyyMMdd"
    if ($mappingNeedsRefresh -and $Compact -eq $TodayCompact) {
        $code = Invoke-DailyPython "concept_mapping" @("concept_hotspot.py", "--fetch-mapping")
        if ($code -ne 0) { Write-Host "  [WARN] concept mapping refresh failed; hotspot members remain usable." -ForegroundColor Yellow }
    }
    elseif ($mappingNeedsRefresh) {
        Write-Host "  [WARN] mapping is stale; historical runs do not fetch future membership." -ForegroundColor Yellow
        $script:RunIssues += "Concept mapping is stale; historical runs cannot refresh current membership."
    }
    else {
        Write-Host "  concept mapping is fresh." -ForegroundColor Green
    }

    Write-Host "`n[6/6] concept hotspot..." -ForegroundColor Cyan
    $code = Invoke-DailyPython "concept_hotspot" @("concept_hotspot.py", "--fetch-hotspot", "--date", $Compact)
    if ($code -ne 0) { Write-Host "  [WARN] concept hotspot non-zero exit (API may have changed), continue (pipeline uses cache)." -ForegroundColor Yellow }

    Write-Host "`n[weekly] fundamentals panel (S4 reference ranking)..." -ForegroundColor Cyan
    $finPath = "results\r4_fundamentals_panel.csv"
    $finStale = (-not (Test-Path $finPath)) -or (((Get-Date) - (Get-Item $finPath).LastWriteTime).Days -gt 6)
    if ($finStale -and $Compact -eq $TodayCompact) {
        $code = Invoke-DailyPython "fundamentals_weekly" @("backfill_fundamentals_r4.py", "--recent", "3")
        if ($code -ne 0) {
            Write-Host "  [WARN] fundamentals refresh failed; the S4 reference ranking uses the older panel." -ForegroundColor Yellow
            $script:RunIssues += "Fundamentals weekly refresh failed; S4 reference ranking uses older reports."
        }
    }
    else {
        Write-Host "  fundamentals panel refreshed within 7 days (or historical run); skipped." -ForegroundColor Green
    }
}
else {
    Write-Host "`n(-SkipUpdate set: skipping data update, going straight to analysis)" -ForegroundColor Yellow
}


Write-Host "========== STEP 1.5: POSITION MONITOR (independent of scanning; read-only) ==========" -ForegroundColor Magenta
$monitorExit = Invoke-DailyPython "position_monitor" @("-m", "mainwave_engine.trading.position_monitor", "--date", $Dash)
if ($monitorExit -ne 0) { $script:RunIssues += "Position monitor raised a P0 alert or failed; see results/trading_system/position_monitor/$Compact.md." }
Show-MarketRisk

Write-Host "========== S4 OPPORTUNITIES (before the pipeline so the review can read it; no orders) ==========" -ForegroundColor Magenta
$s4Exit = Invoke-DailyPython "s4_opportunities" @("-m", "mainwave_engine.s4_scan", "--date", $Dash)
if ($s4Exit -ne 0) { $script:RunIssues += "S4 opportunity scan failed; see the step log." }
$s3Exit = Invoke-DailyPython "s3_observations" @("-m", "mainwave_engine.s3_scan", "--date", $Dash)
if ($s3Exit -ne 0) { $script:RunIssues += "S3 observation scan failed; see the step log." }

Write-Host "========== STEP 2: LIFECYCLE DISCOVERY + EXPLAINABLE ANALYSIS ==========" -ForegroundColor Magenta
$mainwaveArgs = @("-m", "mainwave_engine.pipeline", "--date", $Dash)
if ($RebuildMainwave) { $mainwaveArgs += "--rebuild-current" }
if ($RebuildWeeklyTrace) { $mainwaveArgs += "--force-weekly-rebuild" }
$mainwaveExit = Invoke-DailyPython "lifecycle_pipeline" $mainwaveArgs
if ($mainwaveExit -ne 0) {
    Write-Host "[ERROR] lifecycle pipeline failed -> $out\mainwave_run_status.json" -ForegroundColor Red
    Write-Warning "Candidate reports may be old; continuing the independent ETF step."
}
Write-Host "========== STEP 3: ETF CYCLE WATCHLIST (analysis only) ==========" -ForegroundColor Magenta
$etfArgs = @("-m", "mainwave_engine.etf_tracker", "--date", $Dash)
if (-not $SkipUpdate) { $etfArgs += "--refresh" }
$etfExit = Invoke-DailyPython "etf_tracker" $etfArgs
if ($mainwaveExit -ne 0) {
    Write-Host "========== FALLBACK OBSERVATIONS (pipeline failed before review; no orders) ==========" -ForegroundColor Magenta
    $strongExit = Invoke-DailyPython "strong_trend_shadow" @("-m", "mainwave_engine.strong_trend", "--date", $Dash)
    if ($strongExit -ne 0) { Write-Host "  [WARN] strong_trend shadow failed; it never blocks the run." -ForegroundColor Yellow }
    Write-Host "========== S1/S2/S5 OBSERVATIONS + S1-S5 EXPORT (research lists; no orders) ==========" -ForegroundColor Magenta
    $scenarioExit = Invoke-DailyPython "scenario_observations" @("-m", "mainwave_engine.scenario_scan", "--date", $Dash)
    if ($scenarioExit -ne 0) { $script:RunIssues += "S1/S2/S5 scan or the S1-S5 export failed; see the step log." }
}
if ($mainwaveExit -eq 0) {
    Write-Host "========== POSITION MONITOR AFTER CLOSURE VALUATION ==========" -ForegroundColor Magenta
    $monitorFinal = Invoke-DailyPython "position_monitor_final" @("-m", "mainwave_engine.trading.position_monitor", "--date", $Dash)
    if ($monitorFinal -ne 0) { $script:RunIssues += "Position monitor after closure raised a P0 alert; handle positions before new risk." }
}
if ($mainwaveExit -eq 0) {
    Write-Host "========== PREPARE NEXT SESSION'S FAST REFERENCE + POTENTIAL POOL ==========" -ForegroundColor Magenta
    $referenceExit = Invoke-DailyPython "intraday_reference" @("-m", "mainwave_engine.intraday_review", "--date", $Dash, "--prepare")
}
if (Test-Path -LiteralPath "results\intraday\$Compact\latest.json") {
    Write-Host "========== INTRADAY -> OFFICIAL CLOSE RECONCILIATION ==========" -ForegroundColor Magenta
    $settleExit = Invoke-DailyPython "intraday_settlement" @("-m", "mainwave_engine.intraday_review", "--date", $Dash, "--finalize")
    if ($settleExit -ne 0) { $script:RunIssues += "Intraday observations are retained; official takeover is incomplete. Check results/intraday/$Compact/settlement.json." }
}

Write-Host "========== DAILY ANALYSIS SUMMARY ==========" -ForegroundColor Green
Write-Host "Date       : $Dash"
Write-Host "Output     : $out"
Write-Host "Mode       : analysis + isolated paper account if configured; NO LIVE ORDERS"
if ($mainwaveExit -eq 0) {
    try {
        $board = Get-Content "$out\analysis_board.json" -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json
        $status = Get-Content "$out\mainwave_run_status.json" -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json
        if ($board.as_of -ne $Dash -or $status.as_of -ne $Dash -or $status.status -ne "SUCCESS") {
            throw "Analysis date/status does not match this run."
        }
        Write-Host "Engine     : $($status.engine_version) | $($status.status) | steps=$($status.steps.Count)"
        if ($status.delivery_status -eq "DEGRADED") {
            foreach ($warning in $status.warnings) {
                $script:RunIssues += $warning
                Write-Warning $warning
            }
        }
        # Older reports have empty dictionary keys in diagnostic counters, unsupported by PS 5.1.
        # Project only the display contract with Python's JSON parser; leave archived reports intact.
        $readRouteSummary = @'
import json, sys
with open(sys.argv[1], encoding='utf-8-sig') as source:
    report = json.load(source)
fields = ('as_of', 'universe', 'fresh', 'cache_hits', 'routes', 'overlap_stocks', 'extra_route_memberships', 'policy')
print(json.dumps({key: report.get(key) for key in fields}))
'@
        $routeJson = & python -c $readRouteSummary "$out\route_scan_summary.json"
        if ($LASTEXITCODE -ne 0) { throw "Could not read route_scan_summary.json (exit=$LASTEXITCODE)." }
        $routes = $routeJson | ConvertFrom-Json -ErrorAction Stop
        if ($routes.as_of -ne $Dash) { throw "Route summary date mismatch." }
        Write-Host "Universe   : $($routes.universe) | fresh=$($routes.fresh) | cached=$($routes.cache_hits)"
        Write-Host "Routes     : R1=$($routes.routes.R1) R2=$($routes.routes.R2) R3=$($routes.routes.R3) | overlaps allowed, not buy counts"
        Write-Host "Overlap    : stocks=$($routes.overlap_stocks) | extra memberships=$($routes.extra_route_memberships)"
        Write-Host "Data scope : local universe; stale data is not proof of suspension"
        Write-Host "Policy     : $($routes.policy.policy) | $($routes.policy.validation)"
        Write-Host "Market     : $($board.market.structure) / $($board.market.temperature) | data=$($board.market.data_status)"
        $analysisTotal = [int]$board.summary.input_candidates + [int]$board.summary.tracked_extra
        Write-Host "Analysis   : $($board.summary.analyzed)/$analysisTotal | failed=$($board.summary.failed) | version=$($board.version)"
        Write-Host "Coverage   : candidates=$($board.summary.input_candidates) | continued=$($board.summary.tracked_extra) | displayed=$($board.summary.displayed)"
        foreach ($item in $board.summary.advice_counts.PSObject.Properties) {
            Write-Host "  $($item.Name): $($item.Value)"
        }
        Write-Host "Priority   : review categories, not expected-return ranking"
        $board.records | Select-Object -First 5 symbol, name, route_ids, active_strategy, value_tier_label | Format-Table -AutoSize
        $closure = Get-Content "$out\trading_closure_status.json" -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json
        if ($closure.as_of -ne $Dash -or $closure.status -ne "SUCCESS") { throw "Trading closure missing/incomplete for this date." }
        Write-Host "Closure    : candidates=$($closure.ranking.total) | rankable=$($closure.ranking.rankable) | plan_review=$($closure.ranking.paper_plan_eligible)"
        Write-Host "Paper      : $($closure.paper.status) | live_orders=False | separate from legacy virtual observations"
        Write-Host "Validation : mature=$($closure.validation.mature_rows) | pending=$($closure.validation.pending_rows) | data_wait=$($closure.validation.data_wait_rows) | no automatic tuning"
    }
    catch {
        $script:RunIssues += "Analysis summary unavailable: $_"
        Write-Warning $script:RunIssues[-1]
    }
}
Write-Host "ETF        : exit=$etfExit"
try {
    $paperFile = Get-Item -LiteralPath "$out\virtual_positions.json" -ErrorAction Stop
    if ($paperFile.LastWriteTime -lt $runStarted) { throw "Virtual report was not refreshed in this run." }
    $paper = Get-Content $paperFile.FullName -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json
    if ($paper.as_of -ne $Dash) { throw "Virtual report date mismatch." }
    Write-Host "Virtual    : total=$($paper.summary.total) | sell=$($paper.summary.SELL) | reduce=$($paper.summary.REDUCE) | confirm=$($paper.summary.SOFT_CONFIRMING) | data_wait=$($paper.summary.DATA_WAIT)"
    if ($paper.summary.status -ne "SUCCESS") { $script:RunIssues += "Virtual positions: $($paper.summary.status); check missing/stale data." }
    $paper.records | Where-Object { $_.observation.status -in @('SELL', 'REDUCE', 'SOFT_CONFIRMING', 'DATA_WAIT') } | ForEach-Object {
        Write-Host "  $($_.symbol) $($_.name): $($_.observation.status) | $($_.observation.reason)" -ForegroundColor Yellow
    }
}
catch {
    $script:RunIssues += "Virtual position report unavailable: $_"
    Write-Warning $script:RunIssues[-1]
}
Write-Host "Reports:"
Write-Host "  Overview : $out\analysis_board.md"
Write-Host "  Discovery: $out\mainwave_three_direction_report.md"
Write-Host "  R2 watch : $out\r2_stage_watch.md (observation only)"
Write-Host "  Path A/B : $out\mainwave_path_shadow.json (no orders)"
Write-Host "  Routes   : $out\route_scan.md"
Write-Host "  Coverage : $out\route_scan_audit.csv"
Write-Host "  Themes   : $out\theme_horizon.md"
Write-Host "  Lifecycle: $out\stock_lifecycle.csv"
Write-Host "  Audit    : $out\analysis_candidate_audit.csv"
Write-Host "  Decisions: $out\analysis_decision_funnel.csv"
Write-Host "  Evidence : $out\analysis_evidence.json"
Write-Host "  Virtual  : $out\virtual_positions.md"
Write-Host "  S4 picks : $out\s4_opportunities.md (manual: python -m mainwave_engine.s4_scan --date $Dash --pick CODE)"
Write-Host "  S1-S5    : $out\scenario_opportunities_$Compact.csv (export of the day's S1-S5 opportunities)"
Write-Host "  Closure  : $out\trading_closure.md"
Write-Host "  Review   : $out\eod_review\status.json (latest report version and warnings)"
Write-Host "  Relay    : $out\board_relay\status.json (prior-session watch; no orders)"
Write-Host "  Attribution: $out\limit_theme\latest.json (limit-up themes, media, filings and LHB evidence)"
Write-Host "  Ranking  : $out\ranking_validation.json"
Write-Host "  ETF      : results\etf_tracker\$Compact\daily_report.md"
Write-Host "  Full run : $out\daily_full_status.json"
Write-Host "  Health   : $out\mainwave_run_status.json"
Write-Host "Viewer     : python mainwave_visual_app.py -> http://127.0.0.1:8068/"
Write-Host "Trading UI : http://127.0.0.1:8068/trading"
Write-Host "Theme UI   : http://127.0.0.1:8068/limit-themes"

Write-Host "`n========== DAILY CORE FOCUS (observation, not orders) ==========" -ForegroundColor Cyan
if ($mainwaveExit -eq 0) {
    $focusArgs = @("-m", "mainwave_engine.daily_focus", "--date", $Dash)
    if ($s4Exit -ne 0) { $focusArgs += "--skip-s4" }
    $focusExit = Invoke-DailyPython "daily_focus" $focusArgs
    if ($focusExit -ne 0) { $script:RunIssues += "Core focus report not generated for this run; do not reuse older results." }
}
else {
    Write-Warning "Core focus unavailable: the main pipeline failed; older lists are not shown."
}

$failed = @($script:RunSteps | Where-Object { $_.exit_code -ne 0 })
$runState = if ($mainwaveExit -ne 0) { "FAILED" } elseif ($failed.Count -gt 0 -or $script:RunIssues.Count -gt 0) { "PARTIAL" } else { "SUCCESS" }
Write-DailyStatus $runState
Show-MarketRisk
Write-Host "`n========== DAILY RUN: $runState ==========" -ForegroundColor $(if ($runState -eq "SUCCESS") { "Green" } else { "Yellow" })
$script:RunSteps | Select-Object name, status, seconds | Format-Table -AutoSize | Out-Host
foreach ($issue in $script:RunIssues) { Write-Warning $issue }
if ($runState -ne "SUCCESS") {
    Write-Warning "Run incomplete. See failed steps/issues above; rerun daily_full after resolving them."
    if ($failed.Count -gt 0) { exit $failed[0].exit_code }
    exit 1
}
Write-Host "Done. Read the analysis overview; open the viewer for evidence and lifecycle details." -ForegroundColor Green
exit 0
