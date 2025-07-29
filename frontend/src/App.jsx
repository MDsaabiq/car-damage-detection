import { useState, useCallback } from 'react'
import ImageUploader from './components/ImageUploader'
import ResultDisplay from './components/ResultDisplay'
import './App.css'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleImageUpload = useCallback(async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1>🚗 Car Damage Classifier</h1>
        <p>Upload an image to analyze front/rear damage severity</p>
      </header>
      
      <main className="app-main">
        <ImageUploader onImageUpload={handleImageUpload} loading={loading} />
        {error && <div className="error">Error: {error}</div>}
        {result && <ResultDisplay result={result} />}
      </main>
    </div>
  )
}

export default App
