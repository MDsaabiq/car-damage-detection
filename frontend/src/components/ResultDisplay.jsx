import './ResultDisplay.css'

const ResultDisplay = ({ result }) => {
  const getDamageColor = (prediction) => {
    if (prediction.includes('Normal')) return '#22c55e'  // Green for normal
    if (prediction.includes('Crushed')) return '#f59e0b'  // Orange for crushed
    if (prediction.includes('Breakage')) return '#ef4444'  // Red for breakage
    return '#6b7280'  // Gray default
  }

  const getDamageIcon = (prediction) => {
    if (prediction.includes('Normal')) return '✅'
    if (prediction.includes('Crushed')) return '⚠️'
    if (prediction.includes('Breakage')) return '🚨'
    return '❓'
  }

  const getPositionIcon = (prediction) => {
    return prediction.startsWith('F_') ? '🚘' : '🚗'
  }

  const formatPrediction = (prediction) => {
    const position = prediction.startsWith('F_') ? 'Front' : 'Rear'
    const damage = prediction.split('_')[1]
    return `${position} - ${damage}`
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
            {getPositionIcon(result.prediction)} {getDamageIcon(result.prediction)}
          </div>
          <div className="prediction-text">
            <h3 style={{ color: getDamageColor(result.prediction) }}>
              {formatPrediction(result.prediction)}
            </h3>
            <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>

        <div className="probabilities">
          <h4>Detailed Probabilities</h4>
          {Object.entries(result.probabilities).map(([category, probability]) => (
            <div key={category} className="probability-bar">
              <div className="probability-label">
                <span>{getPositionIcon(category)} {getDamageIcon(category)} {formatPrediction(category)}</span>
                <span>{(probability * 100).toFixed(1)}%</span>
              </div>
              <div className="probability-track">
                <div 
                  className="probability-fill"
                  style={{ 
                    width: `${probability * 100}%`,
                    backgroundColor: getDamageColor(category)
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
