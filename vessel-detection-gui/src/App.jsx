import { useState, useEffect, useRef } from 'react'
import VideoDisplay from './components/VideoDisplay'
import ControlPanel from './components/ControlPanel'
import VideoSelector from './components/VideoSelector'
import './App.css'

function App() {
  const [videoPath, setVideoPath] = useState(null)
  const [mode, setMode] = useState(null) // 'all_detections' or 'target_vessels'
  const [selectedTrackIds, setSelectedTrackIds] = useState(new Set())
  const [taggedTrackIds, setTaggedTrackIds] = useState(new Set()) // Track IDs that have been tagged
  const [isTracking, setIsTracking] = useState(false)
  const [isPaused, setIsPaused] = useState(true)
  const [currentFrame, setCurrentFrame] = useState(null)
  const [detections, setDetections] = useState([])
  const [frameCount, setFrameCount] = useState(0)
  const [videoInfo, setVideoInfo] = useState(null)
  const wsRef = useRef(null)

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//localhost:8000/ws`
    
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'frame') {
        setCurrentFrame(data.frame)
        setDetections(data.detections || [])
        setFrameCount(data.frame_count || 0)
      } else if (data.type === 'video_ended') {
        setIsTracking(false)
        setIsPaused(true)
        console.log('Video ended:', data.message)
        alert(`✓ Video playback completed at frame ${data.frame_count}`)
      } else if (data.type === 'auto_play_complete') {
        setIsPaused(true)
        console.log('Auto-play complete:', data.message)
        // Don't show alert, just log - user can see detections are ready
      } else if (data.type === 'error') {
        console.error('WebSocket error:', data.message)
        alert(`Error: ${data.message}`)
      } else if (data.type === 'model_loaded') {
        console.log('YOLO model loaded on', data.device)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }
  }

  const handleVideoLoad = async (path) => {
    try {
      const response = await fetch('http://localhost:8000/api/load-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: path })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to load video')
      }

      const data = await response.json()
      setVideoPath(path)
      setVideoInfo(data)
      setMode(null) // Reset mode - user must choose
      setSelectedTrackIds(new Set())
      setTaggedTrackIds(new Set()) // Reset tagged IDs
      setIsTracking(false)
      setIsPaused(false) // New video = fresh state, not paused
      setDetections([])
      
      // Connect WebSocket after video is loaded
      connectWebSocket()
    } catch (error) {
      console.error('Error loading video:', error)
      alert(`Error loading video: ${error.message}`)
    }
  }

  const handleTag = async () => {
    if (selectedTrackIds.size === 0) {
      alert('Please select at least one vessel by clicking on the bounding boxes')
      return
    }

    try {
      const trackIdsArray = Array.from(selectedTrackIds)
      console.log('Sending TAG request with track_ids:', trackIdsArray)
      
      const response = await fetch('http://localhost:8000/api/tag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track_ids: trackIdsArray })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to tag vessels')
      }

      const result = await response.json()
      console.log('✓ Vessels tagged successfully:', result.selected_track_ids)
      // Update tagged track IDs - these are the ones that can be tracked
      setTaggedTrackIds(new Set(result.selected_track_ids))
      alert(`✓ Tagged ${result.selected_track_ids.length} vessel(s): ${result.selected_track_ids.join(', ')}`)
    } catch (error) {
      console.error('Error tagging vessels:', error)
      alert(`Error tagging vessels: ${error.message}`)
    }
  }

  const handleTrack = async () => {
    if (taggedTrackIds.size === 0) {
      alert('Please tag vessels first by selecting them (click bboxes to turn green) and clicking TAG')
      return
    }

    try {
      console.log('Sending TRACK request...')
      const response = await fetch('http://localhost:8000/api/track', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        const error = await response.json()
        console.error('TRACK error response:', error)
        throw new Error(error.detail || 'Failed to start tracking')
      }

      const result = await response.json()
      setIsTracking(true)
      setIsPaused(false)
      console.log('✓ Tracking started:', result)
      alert(`✓ Tracking started for ${result.selected_track_ids?.length || taggedTrackIds.size} vessel(s)`)
    } catch (error) {
      console.error('Error starting tracking:', error)
      alert(`Error starting tracking: ${error.message}`)
    }
  }

  const handlePause = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        throw new Error('Failed to pause')
      }

      setIsTracking(false)
      setIsPaused(true)
      // Clear tagged IDs when pausing to require re-tagging (only in target_vessels mode)
      if (mode === 'target_vessels') {
        setTaggedTrackIds(new Set())
      }
      console.log('Tracking paused')
    } catch (error) {
      console.error('Error pausing:', error)
      alert(`Error pausing: ${error.message}`)
    }
  }

  const handleAllDetections = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/all-detections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to start all detections mode')
      }

      setMode('all_detections')
      setIsTracking(true)
      setIsPaused(false)
      console.log('All detections mode started')
    } catch (error) {
      console.error('Error starting all detections:', error)
      alert(`Error: ${error.message}`)
    }
  }

  const handleTargetVessels = () => {
    setMode('target_vessels')
    setIsTracking(false)
    setIsPaused(true)
    console.log('Target vessels mode selected')
  }

  const handleBboxClick = (trackId) => {
    setSelectedTrackIds(prev => {
      const newSet = new Set(prev)
      if (newSet.has(trackId)) {
        newSet.delete(trackId)
      } else {
        newSet.add(trackId)
      }
      return newSet
    })
  }

  // Extract video filename from path
  const getVideoName = () => {
    if (!videoPath) return null
    const parts = videoPath.split('/')
    return parts[parts.length - 1]
  }

  const videoName = getVideoName()

  return (
    <div className="app">
      <div className="app-header">
        <div className="header-left">
          <h1>Vessel Detection System</h1>
          {videoName && (
            <div className="video-name-display">
              <span className="video-name-label">Video:</span>
              <span className="video-name-value">{videoName}</span>
            </div>
          )}
        </div>
        {videoInfo && (
          <div className="video-info">
            <span>FPS: {videoInfo.fps}</span>
            <span>Resolution: {videoInfo.width}x{videoInfo.height}</span>
            <span>Frame: {frameCount}</span>
          </div>
        )}
      </div>

      <div className="app-content">
      <div className="sidebar">
        <VideoSelector onVideoSelect={handleVideoLoad} />
        <ControlPanel
          mode={mode}
          onAllDetections={handleAllDetections}
          onTargetVessels={handleTargetVessels}
          onTag={handleTag}
          onTrack={handleTrack}
          onPause={handlePause}
          isTracking={isTracking}
          isPaused={isPaused}
          selectedCount={selectedTrackIds.size}
          taggedCount={taggedTrackIds.size}
        />
      </div>

        <div className="main-content">
          <VideoDisplay
            frame={currentFrame}
            detections={detections}
            selectedTrackIds={selectedTrackIds}
            onBboxClick={mode === 'target_vessels' ? handleBboxClick : null}
            width={videoInfo?.width}
            height={videoInfo?.height}
            mode={mode}
          />
        </div>
      </div>
    </div>
  )
}

export default App

