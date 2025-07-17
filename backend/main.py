import React, { useState, useEffect, useCallback } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';
import { ArrowUp, ArrowDown, TrendingUp, HelpCircle, Rss, Clock, BrainCircuit, BarChart, CheckCircle, XCircle, MinusCircle } from 'lucide-react';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

// --- Configuration ---
const API_BASE_URL = "https://shib-trading-app.onrender.com"; // Replace with your actual backend URL if different

// --- Helper Components ---
const StatCard = ({ title, value, icon, change, changeColor, isLoading }) => (
    <div className="bg-gray-800 p-4 md:p-6 rounded-2xl shadow-lg border border-gray-700 transform hover:scale-105 transition-transform duration-300">
        <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-gray-400">{title}</p>
            {icon}
        </div>
        {isLoading ? (
            <div className="h-8 bg-gray-700 rounded-md animate-pulse"></div>
        ) : (
            <h3 className="text-2xl md:text-3xl font-bold text-white">{value}</h3>
        )}
        {change && !isLoading && (
            <p className={`text-sm font-medium mt-1 ${changeColor}`}>
                {change}
            </p>
        )}
    </div>
);

const SignalDisplay = ({ signal, reasoning, isLoading, sentimentScore }) => (
    <div className="bg-gray-800 p-6 rounded-2xl shadow-lg border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center"><BrainCircuit className="mr-2 text-indigo-400" /> AI Trade Signal</h3>
        {isLoading ? (
            <div className="space-y-4">
                <div className="h-10 bg-gray-700 rounded-md animate-pulse w-1/2 mx-auto"></div>
                <div className="h-16 bg-gray-700 rounded-md animate-pulse"></div>
            </div>
        ) : (
            <div className="text-center">
                <div className={`inline-flex items-center justify-center px-6 py-3 rounded-full text-2xl font-bold mb-4
                    ${signal === 'LONG' ? 'bg-green-500/20 text-green-400' :
                      signal === 'SHORT' ? 'bg-red-500/20 text-red-400' :
                      'bg-yellow-500/20 text-yellow-400'}`}>
                    {signal === 'LONG' && <TrendingUp className="mr-2" />}
                    {signal === 'SHORT' && <ArrowDown className="mr-2" />}
                    {signal === 'NEUTRAL' && <HelpCircle className="mr-2" />}
                    {signal}
                </div>
                <p className="text-gray-300 text-sm italic px-4">{reasoning}</p>
                {sentimentScore !== null && sentimentScore !== undefined && (
                     <p className="text-xs text-gray-500 mt-3">Sentiment Score Used: {sentimentScore.toFixed(4)}</p>
                )}
            </div>
        )}
    </div>
);

const NewsPanel = ({ news, isLoading }) => (
    <div className="bg-gray-800 p-6 rounded-2xl shadow-lg border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center"><Rss className="mr-2 text-indigo-400" /> Latest Crypto News</h3>
        <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
            {isLoading ? (
                Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="h-16 bg-gray-700 rounded-md animate-pulse"></div>
                ))
            ) : news.length > 0 ? (
                news.map((item, index) => (
                    <a key={index} href={item.url} target="_blank" rel="noopener noreferrer" className="block p-3 bg-gray-700/50 rounded-lg hover:bg-gray-700 transition-colors">
                        <p className="font-semibold text-sm text-gray-200 truncate">{item.title}</p>
                        <p className="text-xs text-gray-400">{item.source}</p>
                    </a>
                ))
            ) : (
                 <p className="text-gray-400 text-center py-8">Could not load news. API key may be missing.</p>
            )}
        </div>
    </div>
);

const PerformancePanel = ({ trades, isLoading }) => {
    const totalTrades = trades.length;
    const wins = trades.filter(t => t.status === 'win').length;
    const accuracy = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(1) : "0.0";

    return (
        <div className="bg-gray-800 p-6 rounded-2xl shadow-lg border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center"><BarChart className="mr-2 text-indigo-400" /> Performance</h3>
            {isLoading ? (
                 <div className="h-24 bg-gray-700 rounded-md animate-pulse"></div>
            ) : (
                <div className="grid grid-cols-2 gap-4 text-center">
                    <div>
                        <p className="text-3xl font-bold text-white">{totalTrades}</p>
                        <p className="text-sm text-gray-400">Total Signals</p>
                    </div>
                    <div>
                        <p className="text-3xl font-bold text-green-400">{accuracy}%</p>
                        <p className="text-sm text-gray-400">Accuracy</p>
                    </div>
                </div>
            )}
        </div>
    );
};

// --- NEW: Trade History Table ---
const TradeHistoryTable = ({ trades, isLoading }) => {
    const getStatusIcon = (status) => {
        switch (status) {
            case 'win': return <CheckCircle className="text-green-500" />;
            case 'loss': return <XCircle className="text-red-500" />;
            case 'pending': return <Clock className="text-yellow-500" />;
            default: return <MinusCircle className="text-gray-500" />;
        }
    };

    return (
        <div className="bg-gray-800 p-4 md:p-6 rounded-2xl shadow-lg border border-gray-700 col-span-1 lg:col-span-2">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Clock className="mr-2 text-indigo-400" /> Recent Signal History
            </h3>
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left text-gray-300">
                    <thead className="text-xs text-gray-400 uppercase bg-gray-700/50">
                        <tr>
                            <th scope="col" className="px-4 py-3">Date</th>
                            <th scope="col" className="px-4 py-3">Type</th>
                            <th scope="col" className="px-4 py-3">Entry Price</th>
                            <th scope="col" className="px-4 py-3">Status</th>
                            <th scope="col" className="px-4 py-3">AI Reasoning</th>
                        </tr>
                    </thead>
                    <tbody>
                        {isLoading ? (
                            Array.from({ length: 5 }).map((_, i) => (
                                <tr key={i} className="border-b border-gray-700">
                                    <td colSpan="5" className="px-4 py-4">
                                        <div className="h-6 bg-gray-700 rounded-md animate-pulse"></div>
                                    </td>
                                </tr>
                            ))
                        ) : trades.length > 0 ? (
                            trades.slice(0, 10).map(trade => (
                                <tr key={trade.id} className="border-b border-gray-700 hover:bg-gray-700/30">
                                    <td className="px-4 py-4 font-medium whitespace-nowrap">
                                        {new Date(trade.timestamp).toLocaleString()}
                                    </td>
                                    <td className="px-4 py-4">
                                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                                            trade.signalType === 'LONG' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                                        }`}>
                                            {trade.signalType}
                                        </span>
                                    </td>
                                    <td className="px-4 py-4">${trade.entryPrice.toFixed(8)}</td>
                                    <td className="px-4 py-4">
                                        <div className="flex items-center space-x-2">
                                            {getStatusIcon(trade.status)}
                                            <span className="capitalize">{trade.status}</span>
                                        </div>
                                    </td>
                                    <td className="px-4 py-4 text-gray-400 italic max-w-xs truncate" title={trade.aiReasoning}>
                                        {trade.aiReasoning}
                                    </td>
                                </tr>
                            ))
                        ) : (
                            <tr>
                                <td colSpan="5" className="text-center py-8 text-gray-500">
                                    No trade history yet. Run some signals to see results here.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};


// --- Main App Component ---
export default function App() {
    const [shibData, setShibData] = useState({});
    const [historicalData, setHistoricalData] = useState({ prices: [] });
    const [aiSignal, setAiSignal] = useState({ signal: 'NEUTRAL', reasoning: 'Click "Find Next Trade" to get an AI analysis.' });
    const [news, setNews] = useState([]);
    const [allTrades, setAllTrades] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isSignalLoading, setIsSignalLoading] = useState(false);
    const [isHistoryLoading, setIsHistoryLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchData = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const [pricesRes, historyRes, newsRes] = await Promise.all([
                fetch(`${API_BASE_URL}/shib-prices`),
                fetch(`${API_BASE_URL}/shib-historical-data`),
                fetch(`${API_BASE_URL}/crypto-news/5`),
            ]);

            if (!pricesRes.ok || !historyRes.ok) throw new Error('Failed to fetch market data.');
            
            const pricesData = await pricesRes.json();
            const historyData = await historyRes.json();
            
            setShibData(pricesData);
            setHistoricalData(historyData);

            if (newsRes.ok) {
                const newsData = await newsRes.json();
                setNews(newsData.news || []);
            } else {
                console.warn("Could not fetch news. API key might be missing on the backend.");
                setNews([]);
            }

        } catch (err) {
            setError(err.message);
            console.error("Fetch Data Error:", err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const fetchAllTrades = useCallback(async () => {
        setIsHistoryLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/get-all-trades`);
            if (!res.ok) throw new Error('Failed to fetch trade history.');
            const data = await res.json();
            setAllTrades(data.trades || []);
        } catch (err) {
            console.error("Fetch Trades Error:", err);
            setAllTrades([]); // Clear on error
        } finally {
            setIsHistoryLoading(false);
        }
    }, []);


    useEffect(() => {
        fetchData();
        fetchAllTrades();
    }, [fetchData, fetchAllTrades]);

    const handleFindTrade = async () => {
        if (!shibData.current_price || historicalData.prices.length === 0) {
            alert("Market data not loaded yet. Please wait.");
            return;
        }
        setIsSignalLoading(true);
        setError(null);
        try {
            const requestBody = {
                current_price: shibData.current_price,
                price_change_24h: shibData.price_change_24h,
                historical_prices: historicalData.prices.map(p => p[1]),
                strategy_name: "default",
                sentiment_filter_enabled: true
            };

            const res = await fetch(`${API_BASE_URL}/ai-trade-signal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });

            if (!res.ok) {
                 const errData = await res.json();
                 throw new Error(errData.detail || `AI Signal Error: ${res.statusText}`);
            }

            const signalData = await res.json();
            setAiSignal({
                signal: signalData.signal_type,
                reasoning: signalData.reasoning,
                sentimentScore: signalData.sentiment_score
            });

            // If we get a valid signal (not NEUTRAL), save it and refresh history
            if (signalData.signal_type !== 'NEUTRAL') {
                await saveTrade(signalData);
                await fetchAllTrades(); // Refresh history
            }

        } catch (err) {
            setError(err.message);
            console.error("Find Trade Error:", err);
            setAiSignal({ signal: 'ERROR', reasoning: `Failed to get signal: ${err.message}` });
        } finally {
            setIsSignalLoading(false);
        }
    };

    const saveTrade = async (signalData) => {
        // This function would be expanded to calculate TP/SL etc.
        // For now, it saves the basic signal as a pending trade.
        const trade = {
            id: Date.now(),
            signalType: signalData.signal_type,
            entryPrice: shibData.current_price,
            takeProfitPrice: shibData.current_price * (signalData.signal_type === 'LONG' ? 1.02 : 0.98), // Simplified 2% TP
            stopLossPrice: shibData.current_price * (signalData.signal_type === 'LONG' ? 0.99 : 1.01), // Simplified 1% SL
            positionSize: 1000, // Example size
            status: 'pending',
            outcomePrice: null,
            profitLoss: null,
            timestamp: Date.now(),
            aiReasoning: signalData.reasoning,
            sentimentScore: signalData.sentiment_score
        };

        try {
            await fetch(`${API_BASE_URL}/save-trade`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(trade),
            });
        } catch (err) {
            console.error("Failed to save trade:", err);
        }
    };

    const chartData = {
        labels: historicalData.prices.map(p => new Date(p[0]).toLocaleDateString()),
        datasets: [{
            label: 'SHIB Price (USD)',
            data: historicalData.prices.map(p => p[1]),
            borderColor: 'rgba(99, 102, 241, 1)',
            backgroundColor: 'rgba(99, 102, 241, 0.2)',
            fill: true,
            tension: 0.4,
            pointRadius: 0,
        }],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { display: false },
            y: {
                ticks: {
                    callback: (value) => `$${value.toFixed(8)}`,
                    color: '#9CA3AF',
                },
                grid: { color: 'rgba(255, 255, 255, 0.1)' }
            }
        }
    };

    return (
        <div className="bg-gray-900 min-h-screen text-white font-sans p-4 sm:p-6 lg:p-8">
            <div className="max-w-7xl mx-auto">
                <header className="mb-8">
                    <h1 className="text-3xl md:text-4xl font-bold text-white">SHIB Trading AI Dashboard</h1>
                    <p className="text-gray-400 mt-1">Real-time analysis and AI-powered signals for Shiba Inu.</p>
                </header>

                {error && (
                    <div className="bg-red-500/20 text-red-300 p-4 rounded-lg mb-6 border border-red-500/30">
                        <strong>Error:</strong> {error}
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                    <StatCard title="Current Price" value={shibData.current_price ? `$${shibData.current_price.toFixed(8)}` : '...'} icon={<TrendingUp className="text-green-400" />} isLoading={isLoading} />
                    <StatCard title="24h Change" value={shibData.price_change_24h ? `${shibData.price_change_24h.toFixed(2)}%` : '...'} icon={shibData.price_change_24h > 0 ? <ArrowUp className="text-green-400" /> : <ArrowDown className="text-red-400" />} changeColor={shibData.price_change_24h > 0 ? 'text-green-400' : 'text-red-400'} isLoading={isLoading} />
                    <StatCard title="Market Cap" value={shibData.market_cap ? `$${(shibData.market_cap / 1e9).toFixed(2)}B` : '...'} isLoading={isLoading} />
                    <StatCard title="24h Volume" value={shibData.total_volume ? `$${(shibData.total_volume / 1e6).toFixed(2)}M` : '...'} isLoading={isLoading} />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                    <div className="lg:col-span-2 bg-gray-800 p-4 md:p-6 rounded-2xl shadow-lg border border-gray-700">
                        <h3 className="text-lg font-semibold text-white mb-4">30-Day Price Chart</h3>
                        <div className="h-80">
                            {isLoading ? <div className="h-full bg-gray-700 rounded-md animate-pulse"></div> : <Line data={chartData} options={chartOptions} />}
                        </div>
                    </div>
                    <div className="flex flex-col gap-6">
                        <SignalDisplay signal={aiSignal.signal} reasoning={aiSignal.reasoning} isLoading={isSignalLoading} sentimentScore={aiSignal.sentimentScore} />
                        <button
                            onClick={handleFindTrade}
                            disabled={isSignalLoading || isLoading}
                            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50 disabled:cursor-not-allowed text-white font-bold py-3 px-4 rounded-lg transition-all duration-300 flex items-center justify-center shadow-lg hover:shadow-indigo-500/50"
                        >
                            {isSignalLoading ? 'Analyzing...' : 'Find Next Trade'}
                        </button>
                    </div>
                </div>

                 <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                    <PerformancePanel trades={allTrades} isLoading={isHistoryLoading} />
                    <div className="lg:col-span-2">
                        <NewsPanel news={news} isLoading={isLoading} />
                    </div>
                </div>
                
                {/* --- NEWLY ADDED HISTORY TABLE --- */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <TradeHistoryTable trades={allTrades} isLoading={isHistoryLoading} />
                </div>

            </div>
            <footer className="text-center text-gray-500 text-xs mt-12 pb-4">
                <p>Disclaimer: This is a simulation tool for educational purposes only. Not financial advice.</p>
            </footer>
        </div>
    );
}
