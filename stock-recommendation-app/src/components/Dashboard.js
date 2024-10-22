import React, { useState, useEffect } from 'react';
import TickersList from './TickersList';

// Main Dashboard Component
const Dashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [activeTab, setActiveTab] = useState("daily");

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://127.0.0.1:5000/scan/daily`);
      const result = await response.json();
      setData(result);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (err) {
      console.error("Error fetching data:", err);
      setError("Failed to fetch data. Please try again.");
    }
    setLoading(false);
  };

  useEffect(() => {
    if (activeTab !== "tickers") {
      fetchData();
      const intervalId = setInterval(() => {
        fetchData();
      }, 60000);

      return () => clearInterval(intervalId);
    }
  }, [activeTab]);

  const getValue = (obj, path, defaultValue = 0) => {
    try {
      return (
        path.split(".").reduce((acc, part) => acc && acc[part], obj) ??
        defaultValue
      );
    } catch (e) {
      return defaultValue;
    }
  };

  const formatNumber = (number, decimals = 2) => {
    if (number === null || number === undefined) return "0";
    return Number(number).toFixed(decimals);
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div>
          <h2 className="header-title">Stock Trading Dashboard</h2>
          {lastUpdate && activeTab !== "tickers" && (
            <small className="last-update">
              Last updated: {lastUpdate} (Updates every minute)
            </small>
          )}
        </div>
        {activeTab !== "tickers" && (
          <button
            className="refresh-button"
            onClick={() => fetchData()}
          >
            Refresh Now
          </button>
        )}
      </div>

      <div className="tabs-container">
        {["daily", "tickers"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`tab-button ${activeTab === tab ? "active" : ""}`}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === "tickers" ? (
        <TickersList />
      ) : (
        <>
          {loading && <div className="loading-state">Loading...</div>}

          {error && <div className="error-state">{error}</div>}

          <div className="stock-grid">
            {data?.recommendations?.map((stock, index) => (
              <div key={stock.symbol || index} className="stock-card">
                <div className="stock-header">
                  <div>
                    <h3 className="stock-symbol">{stock.symbol}</h3>
                    <p className="stock-price">${formatNumber(stock.price)}</p>
                  </div>
                  <span
                    className={`action-badge ${stock.action.toLowerCase()}`}
                  >
                    {stock.action}
                  </span>
                </div>

                <div className="price-changes">
                  <div>
                    <div className="change-label">Daily Change</div>
                    <div
                      className={`change-value ${
                        getValue(stock, "change_1d", 0) >= 0
                          ? "positive"
                          : "negative"
                      }`}
                    >
                      {formatNumber(getValue(stock, "change_1d"))}%
                    </div>
                  </div>
                </div>

                <div className="signals-container">
                  <div className="signals-title">Signals</div>
                  <div className="signals-grid">
                    {Object.entries(stock.signals || {}).map(([key, value]) => (
                      <div key={key} className="signal-item">
                        <span
                          className={`signal-indicator ${
                            value ? "active" : "inactive"
                          }`}
                        />
                        <span style={{ color: value ? "#000" : "#666" }}>
                          {key.replace(/_/g, " ")}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="metrics-container">
                  <div>
                    <div className="metric-label">RSI</div>
                    <div className="metric-value">
                      {formatNumber(getValue(stock, "rsi"))}
                    </div>
                  </div>
                  <div>
                    <div className="metric-label">Trend</div>
                    <div className="metric-value">
                      {getValue(stock, "trend_strength", 0)}/5
                    </div>
                  </div>
                  <div>
                    <div className="metric-label">Confidence</div>
                    <div className="metric-value">
                      {formatNumber(getValue(stock, "confidence"))}%
                    </div>
                  </div>
                </div>

                <div className="confidence-bar">
                  <div
                    className={`confidence-progress ${
                      getValue(stock, "confidence", 0) > 70
                        ? "confidence-high"
                        : getValue(stock, "confidence", 0) > 40
                        ? "confidence-medium"
                        : "confidence-low"
                    }`}
                    style={{
                      width: `${getValue(stock, "confidence", 0)}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {!loading &&
            !error &&
            (!data?.recommendations || data.recommendations.length === 0) && (
              <div className="empty-state">
                No recommendations available at this time
              </div>
            )}
        </>
      )}
    </div>
  );
};

export default Dashboard;