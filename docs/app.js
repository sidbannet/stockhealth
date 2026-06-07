window.onerror = function(msg, url, lineNo, columnNo, error) {
    console.error("Global Error:", msg, error);
    // Don't show global errors to user once dashboard is loaded
};

let pyodide;
let currentTicker = "";
let currentPrice = 0;
let selectedLegs = []; // Array of { id, type: 'call'|'put', action: 'buy'|'sell', strike, price }

// Utility to format currency
const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);

async function initPyodide() {
    try {
        const loadingText = document.getElementById('loading-text');
        
        loadingText.innerText = "Step 1/4: Loading Pyodide core...";
        pyodide = await loadPyodide();
        
        loadingText.innerText = "Step 2/4: Loading data science packages...";
        await pyodide.loadPackage(['micropip', 'matplotlib', 'pandas', 'scipy', 'lxml', 'beautifulsoup4', 'html5lib', 'pytz', 'ssl', 'scikit-learn']);
        
        loadingText.innerText = "Step 3/4: Installing quantitative dependencies...";
        await pyodide.runPythonAsync(`
import micropip
await micropip.install([
    'multitasking', 'platformdirs', 'peewee', 'pyodide-http', 'requests', 'protobuf', 'websockets',
    'pyluach', 'toolz', 'tzdata', 'korean_lunar_calendar'
])
await micropip.install(['exchange-calendars', 'pandas_market_calendars', 'yfinance'], deps=False)
        `);

        loadingText.innerText = "Step 4/4: Unpacking stockhealth models...";
        const response = await fetch('stockhealth.zip');
        const buffer = await response.arrayBuffer();
        pyodide.unpackArchive(buffer, "zip");
        
        await pyodide.runPythonAsync(`
            # Mock curl_cffi since we bypassed it
            import sys
            from unittest.mock import MagicMock
            sys.modules['curl_cffi'] = MagicMock()
            sys.modules['curl_cffi.requests'] = MagicMock()

            import pyodide_http
            pyodide_http.patch_all()

            # Add CORS proxy to bypass browser restrictions
            import requests
            import urllib.parse
            from requests.models import PreparedRequest
            _orig_get = requests.Session.get
            
            WORKER_URL = 'https://yfinance-proxy.sidban.workers.dev/?url='
            
            def _proxied_get(self, url, **kwargs):
                if url.startswith('https://query') or url.startswith('https://finance'):
                    req = PreparedRequest()
                    req.prepare_url(url, kwargs.get('params', {}))
                    proxy_url = WORKER_URL + urllib.parse.quote(req.url)
                    
                    if 'params' in kwargs:
                        del kwargs['params']
                        
                    return _orig_get(self, proxy_url, **kwargs)
                return _orig_get(self, url, **kwargs)
            requests.Session.get = _proxied_get
            requests.get = lambda url, **kwargs: requests.Session().get(url, **kwargs)

            from stockhealth.analyzer import TimeSeries, Trade
            from stockhealth.simulation import MonteCarlosWithHeston
            import yfinance as yf
            import json
            import numpy as np
            import pandas as pd
            import math
            
            def get_expirations(ticker_sym):
                t = yf.Ticker(ticker_sym)
                try:
                    price = t.fast_info['lastPrice']
                except:
                    price = 0
                return json.dumps({
                    'price': float(price),
                    'options': list(t.options)
                })
                
            def get_option_chain(ticker_sym, date_str):
                t = yf.Ticker(ticker_sym)
                chain = t.option_chain(date_str)
                calls = chain.calls.replace({np.nan: None}).to_dict('records')
                puts = chain.puts.replace({np.nan: None}).to_dict('records')
                return json.dumps({'calls': calls, 'puts': puts})
                
            def run_simulation(ticker_sym, days, legs_json):
                from stockhealth.stock import Stock
                from stockhealth.training import Trends
                from stockhealth.simulation import MonteCarlosWithHeston
                import json
                import numpy as np
                from scipy.stats import gaussian_kde
                
                legs = json.loads(legs_json)
                
                # We need historical data to train the model
                stock = Stock(ticker=ticker_sym, start_date='2020-01-01')
                stock.history()
                
                trends = Trends(stock=stock, number_of_days=days)
                trends.extract_model_features()
                
                sim = MonteCarlosWithHeston(trained_model=trends, number_of_instances=1000, number_of_days=days)
                sim.solve()
                
                # The final simulated prices for the underlying asset
                final_prices = sim.S.iloc[-1].values
                
                # Calculate portfolio payoff at expiration for each simulated price
                payoffs = np.zeros_like(final_prices)
                initial_cost = 0.0
                
                for leg in legs:
                    action_mult = 1 if leg['action'] == 'buy' else -1
                    strike = leg['strike']
                    price = leg['price']
                    
                    initial_cost -= action_mult * price
                    
                    if leg['type'] == 'call':
                        payoffs += action_mult * np.maximum(final_prices - strike, 0)
                    elif leg['type'] == 'put':
                        payoffs += action_mult * np.maximum(strike - final_prices, 0)
                        
                # Total profit/loss per simulation path
                net_pnl = payoffs + initial_cost
                
                # Calculate KDE for the PDF of Profit/Loss
                if np.std(net_pnl) > 1e-6:
                    kde = gaussian_kde(net_pnl)
                    x = np.linspace(np.percentile(net_pnl, 1), np.percentile(net_pnl, 99), 100)
                    y = kde.evaluate(x)
                else:
                    x = [np.mean(net_pnl)]
                    y = [1.0]
                
                # Win probability (percentage of paths > 0)
                win_prob = np.mean(net_pnl > 0) * 100
                expected_value = np.mean(net_pnl)
                
                return json.dumps({
                    'return': float(expected_value * 100), # Using multiplier for contracts
                    'roi': float(win_prob), 
                    'x': x.tolist(),
                    'y': y.tolist()
                })
        `);

        // Hide loading, show dashboard
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');

        // Setup event listeners
        document.getElementById('search-btn').addEventListener('click', handleSearch);

    } catch (error) {
        console.error("Initialization error:", error);
        document.getElementById('loading-text').innerText = "Error initializing engine: " + error.message;
    }
}

async function handleSearch() {
    const ticker = document.getElementById('ticker').value.toUpperCase().trim();
    const errorEl = document.getElementById('asset-error');
    if (!ticker) return;

    errorEl.classList.add('hidden');
    document.getElementById('search-btn').innerText = "Fetching...";
    
    try {
        const jsonStr = await pyodide.runPythonAsync(`get_expirations("${ticker}")`);
        const data = JSON.parse(jsonStr);
        
        if (!data.options || data.options.length === 0) {
            errorEl.innerText = "No options available for this ticker.";
            errorEl.classList.remove('hidden');
            document.getElementById('expirations-section').classList.add('hidden');
            document.getElementById('chain-section').classList.add('hidden');
        } else {
            currentTicker = ticker;
            currentPrice = data.price;
            renderExpirations(data.options);
        }
    } catch (err) {
        console.error("Data Fetch Error:", err);
        errorEl.innerText = "Error fetching data. Check ticker symbol. See console for details.";
        errorEl.classList.remove('hidden');
    }
    
    document.getElementById('search-btn').innerText = "Fetch Expirations";
}

function renderExpirations(dates) {
    const container = document.getElementById('expiration-pills');
    container.innerHTML = '';
    
    dates.forEach((dateStr, idx) => {
        // Calculate rough DTE
        const dte = Math.ceil((new Date(dateStr) - new Date()) / (1000 * 60 * 60 * 24));
        
        const pill = document.createElement('div');
        pill.className = 'pill';
        pill.innerHTML = `${dateStr} <span class="dte">(${dte > 0 ? dte : 0}d)</span>`;
        pill.onclick = () => {
            document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            fetchChain(dateStr);
        };
        container.appendChild(pill);
    });
    
    document.getElementById('expirations-section').classList.remove('hidden');
}

async function fetchChain(dateStr) {
    document.getElementById('chain-section').classList.remove('hidden');
    document.getElementById('spot-price-badge').innerText = `Spot: ${formatCurrency(currentPrice)}`;
    const tbody = document.getElementById('chain-body');
    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;">Loading Chain...</td></tr>';
    
    // Clear legs when chain changes (for simplicity)
    selectedLegs = [];
    updateTray();

    try {
        const jsonStr = await pyodide.runPythonAsync(`get_option_chain("${currentTicker}", "${dateStr}")`);
        const data = JSON.parse(jsonStr);
        renderChain(data.calls, data.puts);
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: red;">Error loading chain</td></tr>`;
        console.error(err);
    }
}

function renderChain(calls, puts) {
    const tbody = document.getElementById('chain-body');
    tbody.innerHTML = '';
    
    // Map puts by strike for easy lookup
    const putsMap = {};
    puts.forEach(p => putsMap[p.strike] = p);
    
    calls.forEach(call => {
        const strike = call.strike;
        const put = putsMap[strike] || {};
        
        const tr = document.createElement('tr');
        
        // Spot price divider logic
        if (currentPrice > 0 && Math.abs(strike - currentPrice) < 1) { // Simplistic divider marker
            const divider = document.createElement('tr');
            divider.className = 'spot-divider';
            divider.innerHTML = `<td colspan="7"></td>`;
            tbody.appendChild(divider);
        }

        const isCallITM = strike < currentPrice;
        const isPutITM = strike > currentPrice;
        
        // Call Vol, Bid, Ask
        tr.innerHTML = `
            <td class="${isCallITM ? 'itm-call' : ''}">${call.impliedVolatility ? (call.impliedVolatility*100).toFixed(1)+'%' : '-'}</td>
            <td class="clickable ${isCallITM ? 'itm-call' : ''}" data-type="call" data-action="sell" data-strike="${strike}" data-price="${call.bid || 0}">${formatCurrency(call.bid || 0)}</td>
            <td class="clickable ${isCallITM ? 'itm-call' : ''}" data-type="call" data-action="buy" data-strike="${strike}" data-price="${call.ask || 0}">${formatCurrency(call.ask || 0)}</td>
            
            <td class="strike-col">${strike.toFixed(1)}</td>
            
            <td class="clickable ${isPutITM ? 'itm-put' : ''}" data-type="put" data-action="sell" data-strike="${strike}" data-price="${put.bid || 0}">${formatCurrency(put.bid || 0)}</td>
            <td class="clickable ${isPutITM ? 'itm-put' : ''}" data-type="put" data-action="buy" data-strike="${strike}" data-price="${put.ask || 0}">${formatCurrency(put.ask || 0)}</td>
            <td class="${isPutITM ? 'itm-put' : ''}">${put.impliedVolatility ? (put.impliedVolatility*100).toFixed(1)+'%' : '-'}</td>
        `;
        tbody.appendChild(tr);
    });

    // Event delegation for clicks
    tbody.querySelectorAll('.clickable').forEach(td => {
        td.addEventListener('click', (e) => {
            const el = e.currentTarget;
            const action = el.dataset.action;
            const type = el.dataset.type;
            const strike = parseFloat(el.dataset.strike);
            const price = parseFloat(el.dataset.price);
            
            // Toggle leg
            const id = `${action}-${type}-${strike}`;
            const existingIdx = selectedLegs.findIndex(l => l.id === id);
            
            if (existingIdx >= 0) {
                selectedLegs.splice(existingIdx, 1);
                el.classList.remove(`leg-selected-${action}`);
            } else {
                selectedLegs.push({ id, action, type, strike, price });
                el.classList.add(`leg-selected-${action}`);
            }
            updateTray();
        });
    });
}

function updateTray() {
    const tray = document.getElementById('strategy-tray');
    const container = document.getElementById('legs-container');
    
    if (selectedLegs.length === 0) {
        tray.classList.add('hidden');
        return;
    }
    
    tray.classList.remove('hidden');
    container.innerHTML = '';
    
    let net = 0;
    
    selectedLegs.forEach(leg => {
        if (leg.action === 'buy') net -= leg.price;
        if (leg.action === 'sell') net += leg.price;
        
        const tag = document.createElement('div');
        tag.className = `leg-tag ${leg.action}`;
        tag.innerHTML = `
            <span>${leg.action === 'buy' ? '+1' : '-1'} ${leg.strike} ${leg.type.toUpperCase()}</span>
            <span>@ ${formatCurrency(leg.price)}</span>
        `;
        container.appendChild(tag);
    });
    
    const costEl = document.getElementById('net-cost');
    costEl.innerText = `Net ${net >= 0 ? 'Credit' : 'Debit'}: ${formatCurrency(Math.abs(net))}`;
    costEl.className = `net-credit-debit ${net >= 0 ? 'net-credit' : 'net-debit'}`;
    
    // Simple Strategy detection
    let name = "Custom Spread";
    if (selectedLegs.length === 1) {
        name = `${selectedLegs[0].action === 'buy' ? 'Long' : 'Short'} ${selectedLegs[0].type === 'call' ? 'Call' : 'Put'}`;
    } else if (selectedLegs.length === 2 && selectedLegs[0].type === selectedLegs[1].type) {
        name = net >= 0 ? "Credit Spread" : "Debit Spread";
    } else if (selectedLegs.length === 4) {
        name = "Iron Condor";
    }
    document.getElementById('strategy-name').innerText = name;
}

// Simulation integration
document.getElementById('run-sim-btn').addEventListener('click', async () => {
    const btn = document.getElementById('run-sim-btn');
    btn.innerText = "Simulating...";
    
    try {
        const legsJson = JSON.stringify(selectedLegs);
        const days = document.getElementById('forecast-days').value;
        const resStr = await pyodide.runPythonAsync(`run_simulation("${currentTicker}", ${days}, '${legsJson}')`);
        const res = JSON.parse(resStr);
        
        document.getElementById('results-section').classList.remove('hidden');
        document.getElementById('metric-return').innerText = `${res.return}%`;
        document.getElementById('metric-roi').innerText = `${res.roi}%`;
        
        Plotly.newPlot('plot-container', [{
            x: res.x,
            y: res.y,
            type: 'scatter',
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.2)',
            line: { color: '#3b82f6' }
        }], {
            title: 'Profit/Loss Probability Density',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' },
            xaxis: { title: 'Return', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'Probability', gridcolor: 'rgba(255,255,255,0.1)' }
        });
        
        // Scroll to results
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        console.error("Simulation failed:", err);
        alert("Simulation failed. Check console for details.");
    }
    
    btn.innerText = "Run Simulation";
});

// Initialize Pyodide on script load
initPyodide();
