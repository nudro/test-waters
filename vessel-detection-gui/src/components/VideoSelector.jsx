import { useState, useEffect } from 'react'
import './VideoSelector.css'

function VideoSelector({ onVideoSelect }) {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadVideos()
  }, [])

  const loadVideos = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/videos')
      if (!response.ok) {
        throw new Error('Failed to load videos')
      }
      const data = await response.json()
      setVideos(data.videos || [])
    } catch (error) {
      console.error('Error loading videos:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSelect = (video) => {
    onVideoSelect(video.path)
  }

  return (
    <div className="video-selector">
      <h2>Select Video</h2>
      {loading ? (
        <div className="loading">Loading videos...</div>
      ) : videos.length === 0 ? (
        <div className="no-videos">No videos found in /media</div>
      ) : (
        <div className="video-list">
          {videos.map((video, index) => (
            <div
              key={index}
              className="video-item"
              onClick={() => handleSelect(video)}
            >
              <div className="video-name">{video.name}</div>
              <div className="video-path">{video.path}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default VideoSelector

