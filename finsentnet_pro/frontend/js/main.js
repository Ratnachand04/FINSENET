/* ═══════════════════════════════════════════
   CONFIGURATION
   ═══════════════════════════════════════════ */
const API_BASE = 'http://localhost:8000/api';

/* ═══════════════════════════════════════════
   APPLICATION STATE
   ═══════════════════════════════════════════ */
const state = {
  capital: 30000,
  riskTolerance: 0.5,
  horizon: '1M',
  currency: 'USD',
  selectedMarkets: [],
  selectedStocks: [],
  analysisResults: null,   // full API / demo result
  currentStockIdx: 0,
  // chart references
  tvChart: null,
  allocChart: null,
  // training + live state
  trainLossChart: null,
  liveRefreshTimer: null,
  livePriceTimer: null,
  currentTimeframe: '1D',
  trainedTickers: {},  // {ticker: true} cache
};

/* ═══════════════════════════════════════════
   DEMO STOCK DATABASE
   ═══════════════════════════════════════════ */
const STOCK_DB = {
  SP500: [
    {ticker:'AAPL',  name:'Apple Inc.',            price:182.52, change:2.31,  sector:'tech'},
    {ticker:'MSFT',  name:'Microsoft Corp',        price:415.20, change:1.12,  sector:'tech'},
    {ticker:'NVDA',  name:'NVIDIA Corporation',    price:875.30, change:4.05,  sector:'tech'},
    {ticker:'GOOGL', name:'Alphabet Inc.',         price:152.40, change:-0.54, sector:'tech'},
    {ticker:'AMZN',  name:'Amazon.com Inc.',       price:185.60, change:1.87,  sector:'tech'},
    {ticker:'META',  name:'Meta Platforms',        price:502.30, change:2.15,  sector:'tech'},
    {ticker:'TSLA',  name:'Tesla Inc.',            price:248.90, change:-1.23, sector:'tech'},
    {ticker:'JPM',   name:'JPMorgan Chase & Co',   price:196.80, change:0.92,  sector:'finance'},
    {ticker:'V',     name:'Visa Inc.',             price:282.40, change:0.67,  sector:'finance'},
    {ticker:'BAC',   name:'Bank of America',       price:38.20,  change:0.55,  sector:'finance'},
    {ticker:'GS',    name:'Goldman Sachs',         price:458.90, change:1.38,  sector:'finance'},
    {ticker:'JNJ',   name:'Johnson & Johnson',     price:152.30, change:-0.34, sector:'health'},
    {ticker:'UNH',   name:'UnitedHealth Group',    price:528.90, change:1.45,  sector:'health'},
    {ticker:'PFE',   name:'Pfizer Inc.',           price:26.80,  change:-0.92, sector:'health'},
    {ticker:'XOM',   name:'Exxon Mobil Corp',      price:104.20, change:0.88,  sector:'energy'},
    {ticker:'CVX',   name:'Chevron Corporation',   price:156.40, change:0.62,  sector:'energy'},
    {ticker:'PG',    name:'Procter & Gamble',      price:162.70, change:0.21,  sector:'consumer'},
    {ticker:'KO',    name:'Coca-Cola Co',          price:60.40,  change:0.33,  sector:'consumer'},
    {ticker:'WMT',   name:'Walmart Inc.',          price:168.20, change:0.76,  sector:'consumer'},
  ],
  NASDAQ: [
    {ticker:'AAPL',  name:'Apple Inc.',            price:182.52, change:2.31,  sector:'tech'},
    {ticker:'MSFT',  name:'Microsoft Corp',        price:415.20, change:1.12,  sector:'tech'},
    {ticker:'NVDA',  name:'NVIDIA Corporation',    price:875.30, change:4.05,  sector:'tech'},
    {ticker:'AMD',   name:'Advanced Micro Devices', price:178.60, change:3.22,  sector:'tech'},
    {ticker:'NFLX',  name:'Netflix Inc.',          price:628.40, change:1.89,  sector:'tech'},
    {ticker:'INTC',  name:'Intel Corporation',     price:30.80,  change:-1.15, sector:'tech'},
    {ticker:'AVGO',  name:'Broadcom Inc.',         price:1380.50,change:2.68,  sector:'tech'},
  ],
  NYSE: [
    {ticker:'JPM',   name:'JPMorgan Chase & Co',   price:196.80, change:0.92,  sector:'finance'},
    {ticker:'BAC',   name:'Bank of America',       price:38.20,  change:0.55,  sector:'finance'},
    {ticker:'WMT',   name:'Walmart Inc.',          price:168.20, change:0.76,  sector:'consumer'},
    {ticker:'DIS',   name:'Walt Disney Co.',       price:112.40, change:0.88,  sector:'consumer'},
  ],
  BSE: [
    {ticker:'RELIANCE.BO', name:'Reliance Industries', price:2840, change:0.82, sector:'energy'},
    {ticker:'TCS.BO',      name:'Tata Consultancy',    price:3920, change:1.15, sector:'tech'},
    {ticker:'HDFCBANK.BO', name:'HDFC Bank',           price:1650, change:0.45, sector:'finance'},
    {ticker:'INFY.BO',     name:'Infosys Ltd.',        price:1580, change:0.92, sector:'tech'},
  ],
  NSE: [
    {ticker:'RELIANCE.NS', name:'Reliance Industries', price:2840, change:0.82, sector:'energy'},
    {ticker:'TCS.NS',      name:'Tata Consultancy',    price:3920, change:1.15, sector:'tech'},
    {ticker:'HDFCBANK.NS', name:'HDFC Bank',           price:1650, change:0.45, sector:'finance'},
    {ticker:'INFY.NS',     name:'Infosys Ltd.',        price:1580, change:0.92, sector:'tech'},
    {ticker:'ICICIBANK.NS',name:'ICICI Bank',          price:1120, change:1.28, sector:'finance'},
    {ticker:'SBIN.NS',     name:'State Bank of India',  price:780,  change:0.55, sector:'finance'},
  ],
  COMMODITIES: [
    {ticker:'GC=F', name:'Gold Futures',  price:2340, change:-0.21, sector:'energy'},
    {ticker:'SI=F', name:'Silver Futures', price:27.80, change:1.05, sector:'energy'},
    {ticker:'CL=F', name:'Crude Oil WTI', price:78.40, change:-0.88, sector:'energy'},
    {ticker:'NG=F', name:'Natural Gas',   price:2.18,  change:2.30,  sector:'energy'},
  ],
  CRYPTO: [
    {ticker:'BTC-USD', name:'Bitcoin',   price:67420, change:3.42, sector:'tech'},
    {ticker:'ETH-USD', name:'Ethereum',  price:3580, change:2.18, sector:'tech'},
    {ticker:'SOL-USD', name:'Solana',    price:148.20, change:5.67, sector:'tech'},
    {ticker:'BNB-USD', name:'BNB',       price:598.40, change:1.32, sector:'tech'},
    {ticker:'XRP-USD', name:'XRP',       price:0.52, change:-0.88, sector:'tech'},
  ],
};

/* ═══════════════════════════════════════════
   UTILITY HELPERS
   ═══════════════════════════════════════════ */
function hashStr(s){let h=0;for(let i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0;}return Math.abs(h);}
function mkRng(seed){return function(){seed=(seed*16807)%2147483647;return(seed-1)/2147483646;};}

function currSym(){
  return {USD:'$',INR:'₹',EUR:'€',GBP:'£'}[state.currency]||'$';
}
function fmt$(v){
  if(v==null) return currSym()+'0';
  return currSym()+Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
}

function getAllStocks(){
  let all = [];
  const markets = state.selectedMarkets.length > 0 ? state.selectedMarkets : ['SP500'];
  markets.forEach(m=>{ if(STOCK_DB[m]) all=all.concat(STOCK_DB[m]); });
  const seen = new Set();
  return all.filter(s=>{ if(seen.has(s.ticker)) return false; seen.add(s.ticker); return true; });
}

/* ═══════════════════════════════════════════
   ANIMATION 1 — PARTICLE NEURAL NETWORK
   ═══════════════════════════════════════════ */
(function(){
  const cv = document.getElementById('particleCanvas');
  const cx = cv.getContext('2d');
  const P = [];
  const N = 75;
  const maxDist = 140;

  function resize(){ cv.width=innerWidth; cv.height=innerHeight; }
  addEventListener('resize', resize); resize();

  for(let i=0;i<N;i++){
    P.push({
      x:Math.random()*cv.width, y:Math.random()*cv.height,
      vx:(Math.random()-0.5)*0.35, vy:(Math.random()-0.5)*0.35,
      r:1.2+Math.random()*1.8, ph:Math.random()*6.28,
    });
  }

  (function loop(){
    cx.clearRect(0,0,cv.width,cv.height);
    const t = performance.now()*0.001;
    // connections
    for(let i=0;i<P.length;i++){
      for(let j=i+1;j<P.length;j++){
        const dx=P[i].x-P[j].x, dy=P[i].y-P[j].y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<maxDist){
          cx.beginPath();
          cx.strokeStyle=`rgba(0,245,255,${(1-d/maxDist)*0.12})`;
          cx.lineWidth=0.5;
          cx.moveTo(P[i].x,P[i].y); cx.lineTo(P[j].x,P[j].y);
          cx.stroke();
        }
      }
    }
    // nodes
    P.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>cv.width)  p.vx*=-1;
      if(p.y<0||p.y>cv.height) p.vy*=-1;
      const glow = 0.35 + 0.3*Math.sin(t*2+p.ph);
      cx.beginPath();
      cx.arc(p.x,p.y,p.r,0,6.28);
      cx.fillStyle=`rgba(0,245,255,${glow})`;
      cx.fill();
    });
    requestAnimationFrame(loop);
  })();
})();

/* ═══════════════════════════════════════════
   ANIMATION 6 — TICKER STRIP
   ═══════════════════════════════════════════ */
(function(){
  const data = [
    {s:'AAPL',p:'$182.52',c:'+2.31%',up:true},  {s:'MSFT',p:'$415.20',c:'+1.12%',up:true},
    {s:'NVDA',p:'$875.30',c:'+4.05%',up:true},   {s:'GOOGL',p:'$152.40',c:'-0.54%',up:false},
    {s:'TSLA',p:'$248.90',c:'-1.23%',up:false},  {s:'AMZN',p:'$185.60',c:'+1.87%',up:true},
    {s:'BTC',p:'$67,420',c:'+3.42%',up:true},    {s:'GOLD',p:'$2,340',c:'-0.21%',up:false},
    {s:'ETH',p:'$3,580',c:'+2.18%',up:true},     {s:'RELIANCE',p:'₹2,840',c:'+0.82%',up:true},
    {s:'NIFTY50',p:'22,480',c:'+1.10%',up:true},  {s:'META',p:'$502.30',c:'+2.15%',up:true},
    {s:'JPM',p:'$196.80',c:'+0.92%',up:true},    {s:'NFLX',p:'$628.40',c:'+1.89%',up:true},
  ];
  const el = document.getElementById('tickerTrack');
  let h='';
  for(let r=0;r<2;r++) data.forEach(t=>{
    h+=`<div class="ticker-item"><span class="symbol">${t.s}</span> <span class="price">${t.p}</span> <span class="${t.up?'up':'down'}">${t.c}</span></div>`;
  });
  el.innerHTML=h;
})();

/* ═══════════════════════════════════════════
   SCREEN NAVIGATION
   ═══════════════════════════════════════════ */
function goToScreen(n){
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  const target = document.getElementById('screen'+n);
  if(target) target.classList.add('active');
  if(n===3) populateStockList();
  if(n===4) renderDashboard();
  if(n===5) renderPortfolioSummary();
  // Stop live updates when navigating away from screen 4
  if(n!==4) stopLiveUpdates();
  scrollTo({top:0,behavior:'smooth'});
}

/* ═══════════════════════════════════════════
   SCREEN 1 — CAPITAL CONFIG
   ═══════════════════════════════════════════ */
function setCapital(val){
  state.capital=val;
  document.getElementById('capitalInput').value=val.toLocaleString();
  document.querySelectorAll('.quick-btn').forEach(b=>b.classList.remove('active'));
  if(event&&event.target) event.target.classList.add('active');
}

function updateCurrency(){
  state.currency=document.getElementById('currencySelect').value;
  document.getElementById('currSymbol').textContent=currSym();
}

function updateRiskLabel(){
  const v=+document.getElementById('riskSlider').value;
  state.riskTolerance=v/10;
  const labels=['Ultra Safe','Very Conservative','Conservative','Mod. Conservative','Moderate',
    'Moderate','Moderate Aggressive','Aggressive','Very Aggressive','Very Aggressive','Ultra Aggressive'];
  document.getElementById('riskLabel').textContent=labels[v];
}

function setHorizon(btn,val){
  state.horizon=val;
  document.querySelectorAll('.horizon-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}

/* ═══════════════════════════════════════════
   FMP API KEY MANAGEMENT
   ═══════════════════════════════════════════ */
function setFmpCountText(count, suffix){
  const label = count + ' key' + (count!==1 ? 's' : '') + (suffix || '');
  document.querySelectorAll('[data-fmp-count]').forEach(el=>{ el.textContent = label; });
}

function setFmpBudgetStats(total, used, source){
  const remaining = total!=null && used!=null ? Math.max(0, total - used) : null;
  const totalEl = document.getElementById('fmpBudgetTotal');
  const usedEl = document.getElementById('fmpBudgetUsed');
  const remEl = document.getElementById('fmpBudgetRemaining');
  const srcEl = document.getElementById('fmpSource');
  if(totalEl) totalEl.textContent = total!=null ? total.toLocaleString() : '—';
  if(usedEl) usedEl.textContent = used!=null ? used.toLocaleString() : '—';
  if(remEl) remEl.textContent = remaining!=null ? remaining.toLocaleString() : '—';
  if(srcEl) srcEl.textContent = source || '—';
}

function saveFmpKeys(){
  const raw = document.getElementById('fmpKeysInput').value.trim();
  if(!raw){ document.getElementById('fmpStatus').textContent='⚠ No keys entered'; return; }
  const keys = raw.split(/[\n,]+/).map(k=>k.trim()).filter(k=>k.length>5);
  if(keys.length===0){ document.getElementById('fmpStatus').textContent='⚠ Invalid keys'; return; }

  document.getElementById('fmpStatus').textContent='Saving...';
  fetch(`${API_BASE}/live/settings/api-keys`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({fmp_keys:keys})
  }).then(r=>r.json()).then(d=>{
    if(d.status==='ok'){
      const cnt = d.sources?.fmp_keys_count||keys.length;
      setFmpCountText(cnt);
      document.getElementById('fmpStatus').textContent=`✅ ${cnt} key(s) saved · ${d.sources?.fmp_daily_budget||cnt*250} calls/day`;
      document.getElementById('fmpKeysInput').value='';
      loadFmpKeyStatus();
    } else {
      document.getElementById('fmpStatus').textContent='⚠ '+JSON.stringify(d);
    }
  }).catch(e=>{
    document.getElementById('fmpStatus').textContent='⚠ Server offline — keys saved locally';
    localStorage.setItem('fmp_keys',JSON.stringify(keys));
    setFmpCountText(keys.length, ' (local)');
    setFmpBudgetStats(keys.length * 250, 0, 'Local');
  });
}

function loadFmpKeyStatus(){
  fetch(`${API_BASE}/live/settings/api-keys`).then(r=>r.json()).then(d=>{
    const src=d.sources||{};
    if(src.fmp){
      const cnt=src.fmp_keys_count||0;
      setFmpCountText(cnt);
      if(src.fmp_status){
        setFmpBudgetStats(src.fmp_status.total_budget, src.fmp_status.total_used, 'Server');
      } else {
        setFmpBudgetStats(src.fmp_daily_budget || cnt * 250, null, 'Server');
      }
      if(src.fmp_status && src.fmp_status.keys){
        let html='<div style="margin-top:6px;">';
        src.fmp_status.keys.forEach(k=>{
          const pct=Math.round((k.calls_used/(k.calls_used+k.calls_remaining))*100)||0;
          const col=k.exhausted?'#ff4444':'var(--green)';
          html+=`<div style="margin:3px 0;display:flex;align-items:center;gap:8px;">
            <span style="color:var(--cyan);">Key #${k.key_index}</span>
            <div style="flex:1;height:4px;background:#1a1a3a;border-radius:2px;overflow:hidden;">
              <div style="width:${pct}%;height:100%;background:${col};"></div>
            </div>
            <span>${k.calls_used}/${k.calls_used+k.calls_remaining}</span>
          </div>`;
        });
        html+=`<div style="margin-top:6px;color:var(--gold);">Total: ${src.fmp_status.total_used}/${src.fmp_status.total_budget} calls used today</div>`;
        html+='</div>';
        document.getElementById('fmpKeyStatus').innerHTML=html;
      } else {
        document.getElementById('fmpKeyStatus').innerHTML='<div style="margin-top:6px;">No key usage telemetry from server.</div>';
      }
    } else {
      setFmpCountText(0);
      setFmpBudgetStats(null, null, 'Server');
      document.getElementById('fmpKeyStatus').innerHTML='<div style="margin-top:6px;">No server keys configured.</div>';
    }
  }).catch(()=>{
    const localRaw = localStorage.getItem('fmp_keys');
    const localKeys = localRaw ? JSON.parse(localRaw) : [];
    setFmpCountText(localKeys.length, localKeys.length ? ' (local)' : '');
    setFmpBudgetStats(localKeys.length * 250, 0, 'Local');
    document.getElementById('fmpKeyStatus').innerHTML=
      localKeys.length ? '<div style="margin-top:6px;">Using locally saved keys. Server offline.</div>' :
      '<div style="margin-top:6px;">Server offline and no local keys found.</div>';
  });
}

// Load FMP status on page load
document.addEventListener('DOMContentLoaded',()=>{ setTimeout(loadFmpKeyStatus,1000); });

/* ═══════════════════════════════════════════
   SCREEN 2 — MARKET SELECTOR
   ═══════════════════════════════════════════ */
function toggleMarket(el,market){
  el.classList.toggle('selected');
  const idx=state.selectedMarkets.indexOf(market);
  if(idx>=0) state.selectedMarkets.splice(idx,1);
  else state.selectedMarkets.push(market);
}

/* ═══════════════════════════════════════════
   SCREEN 3 — STOCK SEARCH + SELECT
   ═══════════════════════════════════════════ */
function populateStockList(){
  const stocks=getAllStocks();
  // trending tags
  document.getElementById('trendingTags').innerHTML=stocks.slice(0,8).map(s=>{
    const cls=s.change>=0?'up':'down';
    return `<div class="trending-tag" onclick="addStock('${s.ticker}')"><span class="sym">${s.ticker}</span> <span class="${cls}">${s.change>=0?'+':''}${s.change}%</span></div>`;
  }).join('');
  renderStockList(stocks);
}

function renderStockList(stocks){
  document.getElementById('stockList').innerHTML=stocks.map(s=>{
    const cls=s.change>=0?'up':'down';
    const sign=s.change>=0?'▲ +':'▼ ';
    const sel=state.selectedStocks.includes(s.ticker);
    return `<div class="stock-row" ondblclick="addStock('${s.ticker}')">
      <div class="info"><span class="ticker">${s.ticker}</span><span class="company">${s.name}</span></div>
      <span class="price ${cls}">${fmt$(s.price)} <small>${sign}${Math.abs(s.change)}%</small></span>
      <button class="add-btn ${sel?'added':''}" onclick="event.stopPropagation();${sel?`removeStock('${s.ticker}')`:`addStock('${s.ticker}')`}">${sel?'✓ ADDED':'+ ADD'}</button>
    </div>`;
  }).join('');
}

function filterStocks(){
  const q=document.getElementById('stockSearch').value.toUpperCase();
  renderStockList(getAllStocks().filter(s=>s.ticker.includes(q)||s.name.toUpperCase().includes(q)));
}

function filterSector(btn,sector){
  document.querySelectorAll('.sector-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  let stocks=getAllStocks();
  if(sector!=='all') stocks=stocks.filter(s=>s.sector===sector);
  renderStockList(stocks);
}

function addStock(ticker){
  if(!state.selectedStocks.includes(ticker)) state.selectedStocks.push(ticker);
  renderSelectedStocks(); renderStockList(getAllStocks());
}

function removeStock(ticker){
  state.selectedStocks=state.selectedStocks.filter(t=>t!==ticker);
  renderSelectedStocks(); renderStockList(getAllStocks());
}

function renderSelectedStocks(){
  document.getElementById('selectedStocks').innerHTML=state.selectedStocks.map(t=>
    `<div class="selected-tag">${t}<span class="remove" onclick="removeStock('${t}')">✕</span></div>`
  ).join('');
  // Enable/disable analyze button
  const btn=document.getElementById('analyzeBtn');
  if(state.selectedStocks.length===0) btn.classList.add('disabled');
  else btn.classList.remove('disabled');
}

function aiSuggest(){
  const stocks=getAllStocks().sort((a,b)=>b.change-a.change).slice(0,4);
  stocks.forEach(s=>addStock(s.ticker));
}

/* ═══════════════════════════════════════════
   ANALYSIS PIPELINE (Training-Aware)
   ═══════════════════════════════════════════ */
async function runAnalysis(){
  if(!state.selectedStocks.length){ alert('Please select at least one stock.'); return; }
  state.capital=parseInt(document.getElementById('capitalInput').value.replace(/[^0-9]/g,''))||30000;
  const market=state.selectedMarkets[0]||'SP500';

  // ── Phase 1: Train models that haven't been trained yet ──
  for(const ticker of state.selectedStocks){
    try {
      const checkRes=await fetch(`${API_BASE}/train/check/${ticker}`);
      const checkData=await checkRes.json();
      if(checkData.is_trained){
        state.trainedTickers[ticker]=true;
        continue;
      }
    } catch(e) {
      console.log(`[FINSENT] Cannot check training for ${ticker}, will use demo.`);
    }

    // Train this ticker
    await trainModel(ticker, market);
  }

  // ── Phase 2: Get predictions from trained models ──
  showLoading();
  try {
    // Try real analysis endpoint first
    const res=await fetch(`${API_BASE}/analyze`,{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        tickers:state.selectedStocks,
        market:market,
        investment_amount:state.capital,
        risk_tolerance:state.riskTolerance,
        currency:state.currency,
        horizon:state.horizon,
      }),
    });
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    state.analysisResults=await res.json();

    // Enrich with live predictions where available
    await enrichWithLivePredictions(market);
  } catch(e) {
    console.log('[FINSENT] Using demo mode:', e.message);
    state.analysisResults=buildDemoResult();
    // Still try live predictions
    await enrichWithLivePredictions(market);
  }

  await sleep(400);
  hideLoading();
  goToScreen(4);

  // Start live updates
  startLiveUpdates();
}

async function trainModel(ticker, market){
  showTrainingOverlay(ticker);
  try {
    // Start training
    await fetch(`${API_BASE}/train/start`,{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        ticker:ticker, market:market,
        epochs:50, period:'5y',
      }),
    });

    // Poll for progress
    let done=false;
    while(!done){
      await sleep(1500);
      try {
        const res=await fetch(`${API_BASE}/train/status/${ticker}`);
        const data=await res.json();
        const progress=data.progress;
        if(!progress) continue;

        updateTrainingOverlay(progress);

        if(progress.status==='completed'){
          state.trainedTickers[ticker]=true;
          done=true;
        } else if(progress.status==='failed'){
          console.warn(`Training failed for ${ticker}: ${progress.message}`);
          done=true;
        }
      } catch(e){ /* keep polling */ }
    }
  } catch(e) {
    console.log(`[FINSENT] Training unavailable for ${ticker}:`, e.message);
  }
  hideTrainingOverlay();
}

async function enrichWithLivePredictions(market){
  if(!state.analysisResults||!state.analysisResults.signals) return;
  for(let i=0;i<state.analysisResults.signals.length;i++){
    const sig=state.analysisResults.signals[i];
    if(!state.trainedTickers[sig.ticker]) continue;
    try {
      const res=await fetch(`${API_BASE}/live/predict/${sig.ticker}?market=${market}&capital=${state.capital}&risk_tolerance=${state.riskTolerance}`);
      if(!res.ok) continue;
      const pred=await res.json();
      if(pred.status!=='success') continue;
      // Merge live prediction data into the signal
      sig.direction=pred.prediction.direction;
      sig.confidence=pred.prediction.confidence;
      sig.predicted_return=pred.prediction.predicted_return;
      sig.entry_price=pred.signal.entry_price;
      sig.target_price=pred.signal.target_price;
      sig.stop_loss=pred.signal.stop_loss;
      sig.risk_reward=pred.signal.risk_reward;
      sig.capital_required=pred.signal.capital_required;
      sig.quantity=pred.signal.quantity;
      sig.time_horizon=pred.signal.time_horizon;
      sig.sentiment_score=pred.analysis.sentiment_score;
      sig.technical_score=pred.analysis.technical_score;
      sig.regime=pred.analysis.regime;
      sig.reasoning=pred.analysis.reasoning;
      sig._live=true;
    } catch(e){ /* use existing signal */ }
  }
}

/* ═══════════════════════════════════════════
   TRAINING OVERLAY UI
   ═══════════════════════════════════════════ */
function showTrainingOverlay(ticker){
  document.getElementById('trainingOverlay').classList.add('active');
  document.getElementById('trainTicker').textContent=ticker;
  document.getElementById('trainStatus').textContent='Initializing neural network…';
  document.getElementById('trainPct').textContent='0%';
  document.getElementById('trainProgressFill').style.width='0%';
  document.getElementById('trainEpoch').textContent='—';
  document.getElementById('trainLoss').textContent='—';
  document.getElementById('trainValLoss').textContent='—';
  document.getElementById('trainAcc').textContent='—';
  // Reset loss chart
  const ctx=document.getElementById('trainLossChart');
  if(ctx&&state.trainLossChart){ state.trainLossChart.destroy(); state.trainLossChart=null; }
}

function updateTrainingOverlay(p){
  document.getElementById('trainTicker').textContent=p.ticker||'—';
  document.getElementById('trainStatus').textContent=p.message||p.status;
  const pct=p.progress_pct||0;
  document.getElementById('trainPct').textContent=pct.toFixed(0)+'%';
  document.getElementById('trainProgressFill').style.width=pct+'%';
  document.getElementById('trainEpoch').textContent=
    p.current_epoch?`${p.current_epoch}/${p.total_epochs}`:'—';
  document.getElementById('trainLoss').textContent=
    p.train_loss?p.train_loss.toFixed(4):'—';
  document.getElementById('trainValLoss').textContent=
    p.val_loss?p.val_loss.toFixed(4):'—';
  document.getElementById('trainAcc').textContent=
    p.val_accuracy?p.val_accuracy.toFixed(1)+'%':'—';

  // Draw mini loss chart
  if(p.history&&p.history.train_loss&&p.history.train_loss.length>1){
    drawTrainLossChart(p.history);
  }
}

function drawTrainLossChart(history){
  const ctx=document.getElementById('trainLossChart');
  if(!ctx||typeof Chart==='undefined') return;

  if(state.trainLossChart){ state.trainLossChart.destroy(); }
  const labels=history.train_loss.map((_,i)=>i+1);

  state.trainLossChart=new Chart(ctx,{
    type:'line',
    data:{
      labels:labels,
      datasets:[
        { label:'Train', data:history.train_loss, borderColor:'#00F5FF', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3 },
        { label:'Val', data:history.val_loss, borderColor:'#FFD700', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3 },
      ],
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ display:true, position:'top', labels:{ color:'#8899AA', font:{size:9}, boxWidth:12 } } },
      scales:{
        x:{ display:false },
        y:{ display:true, ticks:{ color:'#667788', font:{size:8} }, grid:{ color:'rgba(0,245,255,0.05)' } },
      },
    },
  });
}

function hideTrainingOverlay(){
  document.getElementById('trainingOverlay').classList.remove('active');
  if(state.trainLossChart){ state.trainLossChart.destroy(); state.trainLossChart=null; }
}

/* ═══════════════════════════════════════════
   LIVE DATA UPDATES
   ═══════════════════════════════════════════ */
function startLiveUpdates(){
  // Stop any existing timers
  stopLiveUpdates();

  // Show live indicator
  const liveEl=document.getElementById('liveIndicator');
  if(liveEl) liveEl.style.display='inline-flex';

  // Price polling every 30s
  state.livePriceTimer=setInterval(()=>{
    updateLivePrices();
  }, 30000);

  // Fetch news for current stock
  fetchLiveNews();
}

function stopLiveUpdates(){
  if(state.livePriceTimer){ clearInterval(state.livePriceTimer); state.livePriceTimer=null; }
  if(state.liveRefreshTimer){ clearInterval(state.liveRefreshTimer); state.liveRefreshTimer=null; }
}

async function updateLivePrices(){
  const data=state.analysisResults;
  if(!data||!data.signals) return;
  const sig=data.signals[state.currentStockIdx];
  if(!sig) return;
  const market=state.selectedMarkets[0]||'SP500';

  try {
    const res=await fetch(`${API_BASE}/live/quote/${sig.ticker}?market=${market}`);
    if(!res.ok) return;
    const quote=await res.json();
    if(quote.price){
      document.getElementById('dashPrice').textContent=fmt$(quote.price);
      const up=quote.change_pct>=0;
      document.getElementById('dashChange').innerHTML=
        `<span style="color:${up?'var(--signal-buy)':'var(--signal-sell)'}">${up?'▲':'▼'} ${up?'+':''}${quote.change_pct}% (${quote.source||'live'})</span>`;
    }
  } catch(e){ /* silent */ }
}

async function refreshPrediction(){
  const data=state.analysisResults;
  if(!data||!data.signals) return;
  const sig=data.signals[state.currentStockIdx];
  if(!sig) return;
  const market=state.selectedMarkets[0]||'SP500';

  if(!state.trainedTickers[sig.ticker]){
    console.log('Model not trained for', sig.ticker, '— showing demo data');
    return;
  }

  try {
    const res=await fetch(`${API_BASE}/live/predict/${sig.ticker}?market=${market}&capital=${state.capital}&risk_tolerance=${state.riskTolerance}`);
    if(!res.ok) return;
    const pred=await res.json();
    if(pred.status!=='success') return;

    sig.direction=pred.prediction.direction;
    sig.confidence=pred.prediction.confidence;
    sig.predicted_return=pred.prediction.predicted_return;
    sig.entry_price=pred.signal.entry_price;
    sig.target_price=pred.signal.target_price;
    sig.stop_loss=pred.signal.stop_loss;
    sig.risk_reward=pred.signal.risk_reward;
    sig._live=true;

    renderStockView(state.currentStockIdx);
  } catch(e){ console.log('Refresh failed:', e.message); }
}

async function fetchLiveNews(){
  const data=state.analysisResults;
  if(!data||!data.signals) return;
  const sig=data.signals[state.currentStockIdx];
  if(!sig) return;
  const market=state.selectedMarkets[0]||'SP500';

  try {
    const res=await fetch(`${API_BASE}/live/news/${sig.ticker}?market=${market}`);
    if(!res.ok) return;
    const newsData=await res.json();
    if(newsData.articles&&newsData.articles.length){
      const panel=document.getElementById('newsPanel');
      panel.style.display='block';
      document.getElementById('newsList').innerHTML=newsData.articles.slice(0,6).map(a=>
        `<div class="news-item">
          <div class="news-title">${a.title}</div>
          <div class="news-meta">${a.source||''}${a.published_at?' · '+new Date(a.published_at).toLocaleString():''}</div>
        </div>`
      ).join('');
    }
  } catch(e){ /* news not available */ }
}

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

/* ═══════════════════════════════════════════
   DEMO SIGNAL GENERATOR
   ═══════════════════════════════════════════ */
function buildDemoResult(){
  const signals = state.selectedStocks.map(t=>demoSignal(t));
  const deployed = signals.reduce((s,sig)=>s+sig.capital_required,0);
  return {
    status:'success',
    signals,
    portfolio:{
      allocation:{
        total_deployed:+deployed.toFixed(2),
        cash_remaining:+(state.capital-deployed).toFixed(2),
        utilization:+((deployed/state.capital)*100).toFixed(1),
      }
    },
    risk:{
      sharpe_ratio:  +(1.1+Math.random()*1.8).toFixed(2),
      sortino_ratio: +(1.4+Math.random()*1.6).toFixed(2),
      calmar_ratio:  +(1.2+Math.random()*1.5).toFixed(2),
      max_drawdown:  +(-3-Math.random()*9).toFixed(1),
      var_95:        +(2+Math.random()*5).toFixed(1),
      annualized_return:    +(8+Math.random()*18).toFixed(1),
      annualized_volatility:+(7+Math.random()*10).toFixed(1),
      win_rate: +(52+Math.random()*18).toFixed(1),
    },
  };
}

function demoSignal(ticker){
  const rng=mkRng(hashStr(ticker)+7);
  const pUp = 0.28 + rng()*0.58;
  const dirs=['STRONG SELL','SELL','HOLD','BUY','STRONG BUY'];
  const di = pUp<0.22?0 : pUp<0.38?1 : pUp<0.52?2 : pUp<0.68?3 : 4;
  const db=getAllStocks();
  const info=db.find(s=>s.ticker===ticker)||{price:100,name:ticker};
  const price=info.price;
  const predRet=(rng()-0.3)*0.16;
  const atr=price*0.022;
  const isBuy=di>=3;
  const target=isBuy ? price*(1+Math.abs(predRet)) : price*(1-Math.abs(predRet));
  const stop=isBuy ? price-atr*2 : price+atr*2;
  const rr=Math.abs(target-price)/Math.max(Math.abs(price-stop),0.01);
  const kelly=Math.max(0,Math.min(0.25,(rr*pUp-(1-pUp))/rr)) * (0.5+state.riskTolerance);
  const qty=Math.max(1,Math.floor(state.capital*kelly/price));
  const sent=35+rng()*55;
  const tech=35+rng()*55;
  const regimes=['BULL — Low Vol','BULL — Normal','TRANSITIONAL','BEAR — High Vol','VOLATILE'];
  const regime=regimes[Math.floor(rng()*regimes.length)];

  return {
    ticker, name:info.name,
    direction: dirs[di],
    confidence: +(pUp*100).toFixed(1),
    entry_price: +price.toFixed(2),
    target_price: +target.toFixed(2),
    stop_loss: +stop.toFixed(2),
    risk_reward: +rr.toFixed(2),
    kelly_fraction: +kelly.toFixed(4),
    quantity: qty,
    capital_required: +(qty*price).toFixed(2),
    predicted_return: +(predRet*100).toFixed(2),
    predicted_downside: +(-Math.abs(predRet)*55).toFixed(2),
    time_horizon: rr>2?'3-6 weeks':rr>1?'1-2 weeks':'2-5 days',
    regime,
    sentiment_score: +sent.toFixed(1),
    technical_score: +tech.toFixed(1),
    fusion_confidence: +(pUp*90+5).toFixed(1),
    reasoning:[
      `FINSENT fusion confidence: ${(pUp*100).toFixed(1)}% — ${dirs[di]}`,
      `Predicted return: ${predRet>0?'+':''}${(predRet*100).toFixed(2)}% over ${rr>2?'3-6 weeks':'1-2 weeks'}`,
      `Sentiment analysis (FinBERT): ${sent.toFixed(0)}/100 — ${sent>60?'Positive':'Neutral'} news flow`,
      `Technical score: ${tech.toFixed(0)}/100 — RSI(${(30+rng()*40).toFixed(0)}), ${pUp>0.5?'bullish':'bearish'} MACD crossover`,
      `Regime detection: ${regime}`,
      `Kelly-optimal position: ${kelly.toFixed(2)}% of capital → ${qty} shares`,
      `Risk/Reward: 1:${rr.toFixed(1)} — ${rr>2?'Favorable':'Moderate'} expected payoff`,
    ],
  };
}

/* ═══════════════════════════════════════════
   ANIMATION 12 — LOADING SCAN
   ═══════════════════════════════════════════ */
let loadingInterval = null;
function showLoading(){
  const ov=document.getElementById('loadingOverlay');
  ov.classList.add('active');
  const steps=[...document.querySelectorAll('#loadingSteps li')];
  steps.forEach(s=>s.classList.remove('active','done'));
  let i=0;
  loadingInterval=setInterval(()=>{
    if(i>0 && steps[i-1]) { steps[i-1].classList.remove('active'); steps[i-1].classList.add('done'); }
    if(i<steps.length) steps[i].classList.add('active');
    i++;
    if(i>steps.length) clearInterval(loadingInterval);
  }, 350);
}
function hideLoading(){
  clearInterval(loadingInterval);
  document.getElementById('loadingOverlay').classList.remove('active');
}

/* ═══════════════════════════════════════════
   SCREEN 4 — DASHBOARD RENDERER
   ═══════════════════════════════════════════ */
function renderDashboard(){
  const data=state.analysisResults;
  if(!data||!data.signals||!data.signals.length) return;

  // Stock tabs
  document.getElementById('stockTabs').innerHTML=data.signals.map((s,i)=>
    `<button class="stock-tab ${i===0?'active':''}" onclick="switchStock(${i})">${s.ticker}</button>`
  ).join('');

  state.currentStockIdx=0;
  renderStockView(0);
}

function switchStock(idx){
  state.currentStockIdx=idx;
  document.querySelectorAll('.stock-tab').forEach((t,i)=>t.classList.toggle('active',i===idx));
  renderStockView(idx);
  fetchLiveNews();
  updateLivePrices();
}

function renderStockView(idx){
  const sig=state.analysisResults.signals[idx];
  if(!sig) return;

  /* ── Header ── */
  document.getElementById('dashTicker').textContent=sig.ticker;
  document.getElementById('dashName').textContent=sig.name||'';
  document.getElementById('dashPrice').textContent=fmt$(sig.entry_price);
  const up=sig.predicted_return>=0;
  document.getElementById('dashChange').innerHTML=
    `<span style="color:${up?'var(--signal-buy)':'var(--signal-sell)'}">${up?'▲':'▼'} ${up?'+':''}${sig.predicted_return}% predicted${sig._live?' (LIVE)':''}</span>`;

  /* ── Live indicator ── */
  const liveEl=document.getElementById('liveIndicator');
  if(liveEl) liveEl.style.display=sig._live?'inline-flex':'none';

  /* ── Signal Badge ── */
  const badge=document.getElementById('signalBadge');
  const cls=sig.direction.includes('BUY')?'buy':sig.direction.includes('SELL')?'sell':'hold';
  badge.className='signal-badge '+cls;
  const emoji=cls==='buy'?'🟢':cls==='sell'?'🔴':'🟡';
  document.getElementById('signalDir').textContent=emoji+' '+sig.direction;
  animateCount('signalConf',0,sig.confidence,'%',1200);

  /* ── Trade Levels ── */
  document.getElementById('lvlEntry').textContent=fmt$(sig.entry_price);
  document.getElementById('lvlTarget').textContent=fmt$(sig.target_price);
  document.getElementById('lvlStop').textContent=fmt$(sig.stop_loss);
  document.getElementById('lvlRR').textContent='1 : '+sig.risk_reward.toFixed(1);
  document.getElementById('lvlHorizon').textContent=sig.time_horizon;
  document.getElementById('lvlDeploy').textContent=
    `${sig.quantity} × ${fmt$(sig.entry_price)} = ${fmt$(sig.capital_required)}`;

  /* ── Score Bars (animated) ── */
  requestAnimationFrame(()=>{
    animateBar('barSentiment','valSentiment',sig.sentiment_score);
    animateBar('barTechnical','valTechnical',sig.technical_score);
    animateBar('barFusion','valFusion',sig.fusion_confidence||sig.confidence*0.9);
  });

  /* ── Regime ── */
  const regEl=document.getElementById('regimeBadge');
  regEl.textContent=sig.regime;
  regEl.className='regime-badge '+(
    sig.regime.includes('BULL')?'bull':
    sig.regime.includes('BEAR')?'bear':
    sig.regime.includes('VOLATILE')?'volatile':'transitional'
  );

  /* ── Predictions ── */
  const predR=document.getElementById('predReturn');
  predR.textContent=(sig.predicted_return>=0?'+':'')+sig.predicted_return+'%';
  predR.style.color=sig.predicted_return>=0?'var(--signal-buy)':'var(--signal-sell)';
  document.getElementById('predDownside').textContent=sig.predicted_downside+'%';
  document.getElementById('predDownside').style.color='var(--signal-sell)';

  /* ── Reasoning ── */
  document.getElementById('reasoningList').innerHTML=
    (sig.reasoning||[]).map(r=>`<li>${r}</li>`).join('');

  /* ── Allocation Chart ── */
  renderAllocChart();

  /* ── Risk Meters ── */
  renderRiskMeters();

  /* ── TradingView Chart ── */
  renderTVChart(sig);
}

/* ═══════════════════════════════════════════
   ANIMATION 9 — COUNTER ANIMATION
   ═══════════════════════════════════════════ */
function animateCount(elId,from,to,suffix,dur){
  const el=document.getElementById(elId);
  if(!el) return;
  const start=performance.now();
  (function tick(now){
    const t=Math.min((now-start)/dur,1);
    const ease=1-Math.pow(1-t,3);
    el.textContent=(from+(to-from)*ease).toFixed(1)+suffix;
    if(t<1) requestAnimationFrame(tick);
  })(start);
}

function animateBar(barId,valId,value){
  const bar=document.getElementById(barId);
  if(bar) bar.style.width=value+'%';
  animateCount(valId,0,value,'/100',1600);
}

/* ═══════════════════════════════════════════
   ANIMATION 4 — TRADINGVIEW CHART DRAW (Live Data)
   ═══════════════════════════════════════════ */
async function renderTVChart(sig){
  const container=document.getElementById('chartContainer');
  container.innerHTML='';

  if(typeof LightweightCharts==='undefined'){
    container.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#667;">Loading chart library…</div>';
    return;
  }

  const chart=LightweightCharts.createChart(container,{
    width:container.clientWidth,
    height:400,
    layout:{ background:{color:'transparent'}, textColor:'#8899AA', fontFamily:'JetBrains Mono' },
    grid:{ vertLines:{color:'rgba(0,245,255,0.025)'}, horzLines:{color:'rgba(0,245,255,0.025)'} },
    crosshair:{ mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale:{ borderColor:'rgba(0,245,255,0.08)' },
    timeScale:{ borderColor:'rgba(0,245,255,0.08)', timeVisible:true },
  });

  // Try fetching real candle data
  let candles=null;
  let overlays=null;
  let volumeData=null;
  const market=state.selectedMarkets[0]||'SP500';
  const tf=state.currentTimeframe||'1D';

  try {
    if(tf==='1D'){
      // Daily candles
      const res=await fetch(`${API_BASE}/live/daily/${sig.ticker}?market=${market}&period=6mo`);
      if(res.ok){
        const data=await res.json();
        if(data.candles&&data.candles.length>5){
          candles=data.candles;
          overlays=data.overlays;
          volumeData=data.volume;
        }
      }
    } else {
      // Intraday candles
      const intervalMap={'1m':'1m','5m':'5m','15m':'15m','1H':'1h','4H':'1h'};
      const periodMap={'1m':'1d','5m':'5d','15m':'5d','1H':'5d','4H':'5d'};
      const interval=intervalMap[tf]||'5m';
      const period=periodMap[tf]||'5d';
      const res=await fetch(`${API_BASE}/live/candles/${sig.ticker}?market=${market}&interval=${interval}&period=${period}`);
      if(res.ok){
        const data=await res.json();
        if(data.candles&&data.candles.length>5){
          candles=data.candles;
        }
      }
    }
  } catch(e){
    console.log('[FINSENT] Using demo candles:', e.message);
  }

  // Fallback to generated demo candles
  if(!candles||candles.length<5){
    candles=genCandles(sig.entry_price,120);
  }

  const cs=chart.addCandlestickSeries({
    upColor:'#00FF88', downColor:'#FF3366',
    borderUpColor:'#00FF88', borderDownColor:'#FF3366',
    wickUpColor:'#00FF8880', wickDownColor:'#FF336680',
  });
  cs.setData(candles);

  // Add indicator overlays (SMA, Bollinger Bands)
  if(overlays){
    if(overlays.sma_50&&overlays.sma_50.length){
      const sma50=chart.addLineSeries({color:'#FFD700',lineWidth:1,title:'SMA 50',priceLineVisible:false});
      sma50.setData(overlays.sma_50);
    }
    if(overlays.sma_200&&overlays.sma_200.length){
      const sma200=chart.addLineSeries({color:'#8B5CF6',lineWidth:1,title:'SMA 200',priceLineVisible:false});
      sma200.setData(overlays.sma_200);
    }
    if(overlays.bb_upper&&overlays.bb_upper.length){
      const bbUp=chart.addLineSeries({color:'rgba(0,245,255,0.3)',lineWidth:1,lineStyle:2,priceLineVisible:false});
      bbUp.setData(overlays.bb_upper);
    }
    if(overlays.bb_lower&&overlays.bb_lower.length){
      const bbLow=chart.addLineSeries({color:'rgba(0,245,255,0.3)',lineWidth:1,lineStyle:2,priceLineVisible:false});
      bbLow.setData(overlays.bb_lower);
    }
  }

  // Volume histogram
  const vol=chart.addHistogramSeries({
    priceFormat:{type:'volume'}, priceScaleId:'vol',
  });
  chart.priceScale('vol').applyOptions({scaleMargins:{top:0.85,bottom:0}});

  if(volumeData&&volumeData.length){
    vol.setData(volumeData);
  } else {
    vol.setData(candles.map(c=>({
      time:c.time, value:c.volume||0,
      color: c.close>=c.open ? 'rgba(0,255,136,0.15)' : 'rgba(255,51,102,0.15)',
    })));
  }

  // Price lines: entry, target, stop
  if(sig.entry_price) cs.createPriceLine({price:sig.entry_price, color:'#00F5FF', lineWidth:1, lineStyle:2, title:'Entry'});
  if(sig.target_price) cs.createPriceLine({price:sig.target_price, color:'#00FF88', lineWidth:1, lineStyle:2, title:'Target'});
  if(sig.stop_loss) cs.createPriceLine({price:sig.stop_loss,    color:'#FF3366', lineWidth:1, lineStyle:2, title:'Stop'});

  // Add buy/sell markers on chart if live
  if(sig._live && sig.direction){
    const lastCandle=candles[candles.length-1];
    if(lastCandle){
      const markers=[];
      if(sig.direction.includes('BUY')){
        markers.push({time:lastCandle.time, position:'belowBar', color:'#00FF88', shape:'arrowUp', text:'BUY'});
      } else if(sig.direction.includes('SELL')){
        markers.push({time:lastCandle.time, position:'aboveBar', color:'#FF3366', shape:'arrowDown', text:'SELL'});
      }
      if(markers.length) cs.setMarkers(markers);
    }
  }

  chart.timeScale().fitContent();
  state.tvChart=chart;

  new ResizeObserver(()=>{
    chart.applyOptions({width:container.clientWidth});
  }).observe(container);
}

function genCandles(base,n){
  const out=[]; let p=base*0.92;
  const now=Math.floor(Date.now()/1000);
  for(let i=n;i>0;i--){
    const chg=(Math.random()-0.48)*p*0.026;
    const o=p; p+=chg; const c=p;
    const h=Math.max(o,c)+Math.random()*p*0.012;
    const l=Math.min(o,c)-Math.random()*p*0.012;
    out.push({
      time:now-i*86400,
      open:+o.toFixed(2), high:+h.toFixed(2),
      low:+l.toFixed(2), close:+c.toFixed(2),
      volume:Math.floor(4e6+Math.random()*50e6),
    });
  }
  return out;
}

function setTimeframe(btn,tf){
  document.querySelectorAll('.tf-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  state.currentTimeframe=tf;
  // Re-render chart with new timeframe
  const data=state.analysisResults;
  if(data&&data.signals&&data.signals[state.currentStockIdx]){
    renderTVChart(data.signals[state.currentStockIdx]);
  }
}
function toggleIndicator(btn){
  btn.classList.toggle('active');
  // Re-render chart with toggled indicator
  const data=state.analysisResults;
  if(data&&data.signals&&data.signals[state.currentStockIdx]){
    renderTVChart(data.signals[state.currentStockIdx]);
  }
}

/* ═══════════════════════════════════════════
   ALLOCATION DONUT (Chart.js)
   ═══════════════════════════════════════════ */
function renderAllocChart(){
  const data=state.analysisResults;
  if(!data) return;
  const sigs=data.signals;
  const colors=['#00F5FF','#FFD700','#8B5CF6','#FF6B35','#00FF88','#FF3366','#EC4899','#22C55E'];
  const deployed=sigs.reduce((s,sig)=>s+sig.capital_required,0);
  const cash=Math.max(0,state.capital-deployed);

  const labels=sigs.map(s=>s.ticker).concat(['CASH']);
  const values=sigs.map(s=>s.capital_required).concat([cash]);
  const bg=sigs.map((_,i)=>colors[i%colors.length]).concat(['#28374A']);

  if(state.allocChart) state.allocChart.destroy();
  const ctx=document.getElementById('allocChart').getContext('2d');
  state.allocChart=new Chart(ctx,{
    type:'doughnut',
    data:{ labels, datasets:[{ data:values, backgroundColor:bg, borderWidth:0, hoverOffset:8 }] },
    options:{
      responsive:true, cutout:'68%',
      plugins:{
        legend:{display:false},
        tooltip:{ callbacks:{ label:c=>`${c.label}: ${fmt$(c.raw)} (${((c.raw/state.capital)*100).toFixed(1)}%)` } },
      },
      animation:{ animateRotate:true, duration:1200 },
    },
  });

  document.getElementById('allocList').innerHTML=labels.map((l,i)=>{
    const pct=((values[i]/state.capital)*100).toFixed(1);
    return `<div class="alloc-row"><span><span class="dot" style="background:${bg[i]}"></span>${l}</span><span style="color:${bg[i]}">${pct}%</span></div>`;
  }).join('');
}

/* ═══════════════════════════════════════════
   RISK METERS (ANIMATION 8 — GAUGE SWEEP)
   ═══════════════════════════════════════════ */
function renderRiskMeters(){
  const risk=state.analysisResults.risk||{};
  const meters=[
    {label:'Sharpe',  val:risk.sharpe_ratio||0,         max:4,  color:'var(--cyan)'},
    {label:'Sortino', val:risk.sortino_ratio||0,        max:4,  color:'var(--purple)'},
    {label:'Max DD',  val:Math.abs(risk.max_drawdown||0),max:25, color:'var(--signal-sell)'},
    {label:'VaR 95%', val:risk.var_95||0,               max:15, color:'var(--gold)'},
    {label:'Calmar',  val:risk.calmar_ratio||0,         max:4,  color:'var(--signal-buy)'},
    {label:'Win Rate',val:risk.win_rate||0,             max:100,color:'var(--orange)'},
  ];
  document.getElementById('riskMeters').innerHTML=meters.map(m=>{
    const pct=Math.min((m.val/m.max)*100,100);
    return `<div class="risk-meter">
      <div class="label">${m.label}</div>
      <div class="gauge-mini"><div class="gauge-fill" style="width:0%;background:${m.color};" data-w="${pct}"></div></div>
      <div class="val" style="color:${m.color}">${typeof m.val==='number'?m.val.toFixed(2):m.val}</div>
    </div>`;
  }).join('');
  // Animate gauge fills
  requestAnimationFrame(()=>{
    document.querySelectorAll('.gauge-fill[data-w]').forEach(el=>{
      el.style.width=el.dataset.w+'%';
    });
  });
}

/* ═══════════════════════════════════════════
   SCREEN 5 — PORTFOLIO SUMMARY
   ═══════════════════════════════════════════ */
function renderPortfolioSummary(){
  const data=state.analysisResults;
  if(!data) return;
  const alloc=data.portfolio?.allocation||{};
  const deployed=alloc.total_deployed||data.signals.reduce((s,sig)=>s+sig.capital_required,0);

  document.getElementById('portCapital').textContent=fmt$(state.capital);
  document.getElementById('portDeployed').textContent=fmt$(deployed);
  document.getElementById('portCash').textContent=fmt$(Math.max(0,state.capital-deployed));

  // Table rows
  document.getElementById('portTableBody').innerHTML=data.signals.map(s=>{
    const dc=s.direction.includes('BUY')?'var(--signal-buy)':s.direction.includes('SELL')?'var(--signal-sell)':'var(--signal-hold)';
    const w=((s.capital_required/state.capital)*100).toFixed(1);
    return `<tr>
      <td style="color:var(--cyan);font-weight:600;">${s.ticker}</td>
      <td style="color:${dc};font-weight:600;">${s.direction}</td>
      <td>${s.confidence}%</td>
      <td>${s.quantity}</td>
      <td>${fmt$(s.entry_price)}</td>
      <td style="color:var(--signal-buy)">${fmt$(s.target_price)}</td>
      <td style="color:var(--signal-sell)">${fmt$(s.stop_loss)}</td>
      <td>${fmt$(s.capital_required)}</td>
      <td><span style="color:var(--gold)">${w}%</span></td>
    </tr>`;
  }).join('');

  // Risk summary cards
  const risk=data.risk||{};
  const metrics=[
    {label:'EXPECTED RETURN',      val:(risk.annualized_return||0)+'%', color:'var(--signal-buy)'},
    {label:'SHARPE RATIO',         val:risk.sharpe_ratio||'—',         color:'var(--cyan)'},
    {label:'SORTINO RATIO',        val:risk.sortino_ratio||'—',        color:'var(--purple)'},
    {label:'VOLATILITY',           val:'±'+(risk.annualized_volatility||0)+'%', color:'var(--gold)'},
    {label:'MAX DRAWDOWN',         val:(risk.max_drawdown||0)+'%',     color:'var(--signal-sell)'},
    {label:'VALUE-AT-RISK (95%)',   val:fmt$(state.capital*(risk.var_95||0)/100), color:'var(--orange)'},
  ];
  document.getElementById('riskSummary').innerHTML=metrics.map(m=>
    `<div class="risk-card"><div class="metric-label">${m.label}</div><div class="metric-value" style="color:${m.color}">${m.val}</div></div>`
  ).join('');
}

/* ═══════════════════════════════════════════
   EXPORT REPORT
   ═══════════════════════════════════════════ */
function exportReport(){
  const data=state.analysisResults;
  if(!data) return;

  const line='═'.repeat(55);
  let txt=`${line}\n  FINSENT NET PRO — PORTFOLIO INTELLIGENCE REPORT\n${line}\n\n`;
  txt+=`Capital:        ${fmt$(state.capital)}\n`;
  txt+=`Risk Tolerance: ${(state.riskTolerance*100).toFixed(0)}%\n`;
  txt+=`Horizon:        ${state.horizon}\n`;
  txt+=`Markets:        ${state.selectedMarkets.join(', ')||'SP500'}\n`;
  txt+=`Generated:      ${new Date().toISOString()}\n\n`;

  data.signals.forEach(s=>{
    txt+=`${'─'.repeat(45)}\n`;
    txt+=`  ${s.ticker} — ${s.name||''}\n`;
    txt+=`${'─'.repeat(45)}\n`;
    txt+=`  Signal:     ${s.direction} (${s.confidence}% confidence)\n`;
    txt+=`  Entry:      ${fmt$(s.entry_price)}\n`;
    txt+=`  Target:     ${fmt$(s.target_price)}\n`;
    txt+=`  Stop Loss:  ${fmt$(s.stop_loss)}\n`;
    txt+=`  R/R Ratio:  1:${s.risk_reward}\n`;
    txt+=`  Quantity:   ${s.quantity} shares\n`;
    txt+=`  Deploy:     ${fmt$(s.capital_required)}\n`;
    txt+=`  Horizon:    ${s.time_horizon}\n`;
    txt+=`  Regime:     ${s.regime}\n\n`;
    txt+=`  Reasoning:\n`;
    (s.reasoning||[]).forEach(r=>txt+=`    • ${r}\n`);
    txt+='\n';
  });

  const risk=data.risk||{};
  txt+=`${line}\n  PORTFOLIO RISK METRICS\n${line}\n`;
  txt+=`  Sharpe Ratio:     ${risk.sharpe_ratio}\n`;
  txt+=`  Sortino Ratio:    ${risk.sortino_ratio}\n`;
  txt+=`  Max Drawdown:     ${risk.max_drawdown}%\n`;
  txt+=`  VaR (95%):        ${risk.var_95}%\n`;
  txt+=`  Exp. Return:      ${risk.annualized_return}%\n`;
  txt+=`  Volatility:       ${risk.annualized_volatility}%\n\n`;
  txt+=`${'═'.repeat(55)}\n`;
  txt+=`  Generated by FINSENT NET PRO — AI-Powered Quantitative Trading Intelligence\n`;

  const blob=new Blob([txt],{type:'text/plain'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=`FINSENT_Report_${new Date().toISOString().slice(0,10)}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ═══════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', ()=>{
  updateRiskLabel();
  updateCurrency();
});
