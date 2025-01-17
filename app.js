const chartDiv         = document.getElementById("chart");
const logsDiv          = document.getElementById("logs");
const calcLogsDiv      = document.getElementById("calc-logs");
const ckbBollinger     = document.getElementById("ckb-bollinger");
const ckbEma           = document.getElementById("ckb-ema");
const ckbFib           = document.getElementById("ckb-fib");
const ckbEwt           = document.getElementById("ckb-ewt");
const ckbSupportFlow   = document.getElementById("ckb-support-flow");
const supportPriceInput= document.getElementById("input-support-price");
const inPositionInput  = document.getElementById("input-in-position");

const btnSetTicker     = document.getElementById("btn-set-ticker");
const btnCheckPrice    = document.getElementById("btn-check-price");
const btnCalc          = document.getElementById("calcChart");
const btnClearMarkers  = document.getElementById("btnClearMarkers");
const setChartBtn      = document.getElementById("set-chart-timeframe");

// Ticker & Trade
const tickerInput      = document.getElementById("ticker-input");
const btnPlaceTrade    = document.getElementById("btn-place-trade");
const dailyStartTimeInput = document.getElementById("daily-start-time"); // New Element

/*********************************************
  GLOBALS
*********************************************/
let chart, candleSeries;
let currentTicker = "AAPL";
let currentInterval = "1m";  // Hard-coded to 1 minute
let currentPeriod   = "1d";  // Hard-coded to 1 day
let currentChartData = [];
/*********************************************
  initChart() => Creates a blank chart
*********************************************/
function initChart() {
  chart = LightweightCharts.createChart(chartDiv, {
    width : chartDiv.clientWidth,
    height: chartDiv.clientHeight,
    layout: { backgroundColor: "#ffffff", textColor: "#333" },
    grid: { vertLines: { color: "#eee" }, horzLines: { color: "#eee" } },
    timeScale: { timeVisible: true, secondsVisible: false }
  });
  candleSeries = chart.addCandlestickSeries();

  if (!candleSeries) {
    console.error("Failed to initialize candleSeries.");
    logMessage("Error: Failed to initialize candleSeries.");
  } else {
    console.log("candleSeries initialized.");
    logMessage("Chart initialized.");
  }
}

/*********************************************
  logMessage => writes to #logs
*********************************************/
function logMessage(msg) {
  const time = new Date().toLocaleTimeString();
  logsDiv.textContent += `[${time}] ${msg}\n\n`;
  logsDiv.scrollTop = logsDiv.scrollHeight;
}

/*********************************************
  logCalcMessage => writes to #calc-logs
*********************************************/
function logCalcMessage(msg) {
  const t = new Date().toLocaleTimeString();
  calcLogsDiv.textContent += `[${t}] ${msg}\n`;
  calcLogsDiv.scrollTop = calcLogsDiv.scrollHeight;
}

/*********************************************
  loadChartData => GET /api/fetchData
*********************************************/
async function loadChartData(ticker, interval, period) {
  setChartBtn.disabled = true;

  logMessage(`Fetching chart for ${ticker} - Interval: ${interval}, Period: ${period}`);
  const url = `http://127.0.0.1:5000/api/fetchData?ticker=${ticker}&interval=${interval}&period=${period}`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`HTTP error ${resp.status}`);
    }
    const rawData = await resp.json();
    if (!Array.isArray(rawData) || rawData.length === 0) {
      logMessage(`No data returned for ${ticker} [${interval}, ${period}].`);
      return;
    }

    // Convert data
    let chartData = rawData.map(item => ({
      time : item.time,
      open : item.open,
      high : item.high,
      low  : item.low,
      close: item.close
    }));

    // Overwrite old chart data
    currentChartData = chartData;
    candleSeries.setData(chartData);

    // Calculate min and max for support input
    const minClose = Math.min(...chartData.map(c => c.close));
    const maxClose = Math.max(...chartData.map(c => c.close));
    supportPriceInput.placeholder = `Valid range: ${minClose.toFixed(2)} - ${maxClose.toFixed(2)}`;

    logMessage(`Loaded chart data: ${chartData.length} candles for ${ticker}.`);
  } catch (err) {
    console.error("Error loading chart data:", err);
    logMessage(`Error loading chart data: ${err.message}`);
  } finally {
    setChartBtn.disabled = false;
  }
}

/*********************************************
  runQuantForTimeframe => POST /api/runQuant
*********************************************/
async function runQuantForTimeframe(ticker, interval, period) {
  if (!currentChartData || currentChartData.length === 0) {
    logMessage("No chart data available to run calculations on.");
    return;
  }
  logMessage("Running quant calculations...");

  let supportPrice = null;
  if (ckbSupportFlow.checked) {
    const inputVal = parseFloat(supportPriceInput.value);
    if (!isNaN(inputVal)) {
      supportPrice = inputVal;
      logMessage(`Using user-defined support price: ${supportPrice.toFixed(2)}`);
    } else {
      logMessage("No valid user support price provided => will be auto-calculated.");
    }
  }

  // Get Daily Start Time from the new input
  const dailyStartTime = dailyStartTimeInput.value || "00:00";
  logMessage(`Using Daily Start Time (UTC): ${dailyStartTime}`);

  const payload = {
    ticker,
    theories: {
      bollinger   : ckbBollinger.checked,
      ema         : ckbEma.checked,
      fib         : ckbFib.checked,
      ewt         : ckbEwt.checked,
      support_flow: ckbSupportFlow.checked
    },
    custom_params: {
      support_price: supportPrice,
      user_in_position: parseFloat(inPositionInput.value) || null,
      daily_start_time_utc: dailyStartTime  // Include the daily start time
    },
    client_candles: currentChartData
  };

  try {
    const resp = await fetch("http://127.0.0.1:5000/api/runQuant", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify(payload)
    });
    const result = await resp.json();

    if (!Array.isArray(result.markers)) {
      logMessage(`Quant calc failed: ${result.message || "Unknown error"}`);
      return;
    }

    if (result.markers.length === 0) {
      logMessage(result.message || "No signals generated.");
      return;
    }

    // Apply markers
    candleSeries.setMarkers(result.markers);

    // Log steps if any
    if (Array.isArray(result.calc_log)) {
      result.calc_log.forEach((step) => logCalcMessage(step));
    }

    logMessage(`Applied ${result.markers.length} signals to chart.`);
  } catch (err) {
    logMessage(`Quant calculation error: ${err.message}`);
  }
}

/*********************************************
  fetchLivePrice => GET /api/getPrice
  Then use it to re-fetch the chart
*********************************************/
async function fetchLivePrice(ticker) {
  logMessage(`Checking live price for ${ticker}...`);
  try {
    // Re-load chart data for that ticker (1m,1d)
    await loadChartData(currentTicker, currentInterval, currentPeriod);
    logMessage("Chart refreshed with latest data from yfinance.");
  } catch (err) {
    logMessage(`Failed to refresh chart data: ${err.message}`);
  }
}

/*********************************************
  setInterval => Refresh data automatically
*********************************************/


/*********************************************
  EVENT: Window Resize => adjust chart
*********************************************/
window.addEventListener("resize", () => {
  if (chart) {
    chart.applyOptions({
      width: chartDiv.clientWidth,
      height: chartDiv.clientHeight
    });
  }
});

/*********************************************
  EVENT: SupportFlow => enable/disable input
*********************************************/
ckbSupportFlow.addEventListener("change", () => {
  supportPriceInput.disabled = !ckbSupportFlow.checked;
});

/*********************************************
  EVENT: Ticker => setTicker button
*********************************************/
btnSetTicker.addEventListener("click", async () => {
  currentTicker = tickerInput.value.trim().toUpperCase() || "AAPL";
  logMessage(`Ticker set to => ${currentTicker}`);
  // Clear old markers/log
  candleSeries.setMarkers([]);
  calcLogsDiv.textContent = "";

  // Refresh chart data
  await loadChartData(currentTicker, currentInterval, currentPeriod);
});

/*********************************************
  EVENT: set-chart-timeframe => load 1D/1m
*********************************************/

/*********************************************
  EVENT: calcChart => runQuant
*********************************************/
btnCalc.addEventListener("click", async () => {
  await runQuantForTimeframe(currentTicker, currentInterval, currentPeriod);
});

/*********************************************
  EVENT: checkPrice => refresh chart
*********************************************/
btnCheckPrice.addEventListener("click", async () => {
  await fetchLivePrice(currentTicker);
});

/*********************************************
  EVENT: Clear Markers => remove from chart
*********************************************/
btnClearMarkers.addEventListener("click", () => {
  candleSeries.setMarkers([]);
  logMessage("Cleared all markers from the chart.");
});

/*********************************************
  EVENT: Place Trade => sample usage
*********************************************/
btnPlaceTrade.addEventListener("click", async () => {
  logMessage("Clicked place trade (not implemented).");
});

/*********************************************
  DOMContentLoaded => init chart, load data,
  set up auto-refresh
*********************************************/
document.addEventListener("DOMContentLoaded", () => {
  initChart();
  logMessage("DOM loaded. Chart initialized. Doing initial fetch...");

  // Optionally load initial data
  loadChartData(currentTicker, currentInterval, currentPeriod);

  // Start auto-refresh every 60s
});

/*********************************************
  Block default form submissions
*********************************************/
document.addEventListener("submit", (e) => e.preventDefault());
window.addEventListener("beforeunload", (e) => e.preventDefault());