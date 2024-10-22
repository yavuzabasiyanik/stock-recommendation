import React, { useState, useEffect } from 'react';

// TickersList Component
const TickersList = () => {
  const [tickers, setTickers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchTickers();
  }, []);

  const fetchTickers = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:5000/tickers');
      const data = await response.json();
      setTickers(data.tickers || []);
    } catch (err) {
      setError('Failed to fetch tickers. Please try again.');
      console.error('Error:', err);
    }
    setLoading(false);
  };

  const filteredTickers = tickers.filter(ticker => 
    ticker.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="tickers-container">
      <div className="tickers-header">
        <div>
          <h2>Available Tickers</h2>
          <p className="tickers-count">Total: {tickers.length} tickers</p>
        </div>
        <button className="refresh-button" onClick={fetchTickers}>
          Refresh
        </button>
      </div>

      <input
        type="text"
        className="search-input"
        placeholder="Search tickers..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
      />

      {loading && <div className="loading-state">Loading tickers...</div>}

      {error && <div className="error-state">{error}</div>}

      <div className="tickers-grid">
        {filteredTickers.map((ticker) => (
          <div key={ticker} className="ticker-card">
            <h3 className="ticker-symbol">{ticker}</h3>
          </div>
        ))}
      </div>

      {!loading && !error && filteredTickers.length === 0 && (
        <div className="empty-state">
          No tickers found matching your search.
        </div>
      )}
    </div>
  );
};

export default TickersList;