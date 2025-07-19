import './ResultDisplay.css'

const ResultDisplay = ({ result }) => {
  const getDamageColor = (severity) => {
    switch (severity) {
      case 'normal': return '#22c55e'
      case 'moderate': return '#f59e0b'
      case 'severe': return '#ef4444'
      default: return '#6b7280'
    }
  }

  const getDamageIcon = (severity) => {
    switch (severity) {
      case 'normal': return '✅'
      case 'moderate': return '⚠️'
      case 'severe': return '🚨'
      default: return '❓'
    }
  }

  return (
    <div className="result-container">
      <div className="result-header">
        <h2>Analysis Results</h2>
      </div>
      
      <div className="result-main">
        <div 
          className="prediction-card"
          style={{ borderColor: getDamageColor(result.prediction) }}
        >
          <div className="prediction-icon">
            {getDamageIcon(result.prediction)}
          </div>
          <div className="prediction-text">
            <h3 style={{ color: getDamageColor(result.prediction) }}>
              {result.prediction.toUpperCase()}
            </h3>
            <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>

        <div className="probabilities">
          <h4>Detailed Probabilities</h4>
          {Object.entries(result.probabilities).map(([severity, probability]) => (
            <div key={severity} className="probability-bar">
              <div className="probability-label">
                <span>{getDamageIcon(severity)} {severity}</span>
                <span>{(probability * 100).toFixed(1)}%</span>
              </div>
              <div className="probability-track">
                <div 
                  className="probability-fill"
                  style={{ 
                    width: `${probability * 100}%`,
                    backgroundColor: getDamageColor(severity)
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default ResultDisplay