import { useState, useCallback } from 'react'
import './ImageUploader.css'

const ImageUploader = ({ onImageUpload, loading }) => {
  const [dragActive, setDragActive] = useState(false)
  const [preview, setPreview] = useState(null)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleChange = useCallback((e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }, [])

  const handleFile = useCallback((file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file')
      return
    }

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)

    // Upload file
    onImageUpload(file)
  }, [onImageUpload])

  return (
    <div className="uploader-container">
      <div
        className={`upload-area ${dragActive ? 'drag-active' : ''} ${loading ? 'loading' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          accept="image/*"
          onChange={handleChange}
          disabled={loading}
        />
        
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <div className="preview-overlay">
              {loading ? (
                <div className="spinner">Analyzing...</div>
              ) : (
                <label htmlFor="file-upload" className="upload-button">
                  Upload Different Image
                </label>
              )}
            </div>
          </div>
        ) : (
          <div className="upload-content">
            <div className="upload-icon">📸</div>
            <h3>Drag & Drop Your Car Image</h3>
            <p>or</p>
            <label htmlFor="file-upload" className="upload-button">
              Choose File
            </label>
            <p className="upload-hint">Supports JPG, PNG, WebP</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ImageUploader