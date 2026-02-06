import { useRef, useEffect } from 'react'
import './VideoDisplay.css'

function VideoDisplay({ frame, detections, selectedTrackIds, onBboxClick, width, height, mode }) {
  const canvasRef = useRef(null)
  const imageRef = useRef(null)

  useEffect(() => {
    if (!frame || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()

    img.onload = () => {
      // Set canvas size to match image
      if (width && height) {
        canvas.width = width
        canvas.height = height
      } else {
        canvas.width = img.width
        canvas.height = img.height
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      // Draw bounding boxes with trajectories
      detections.forEach((detection) => {
        const { track_id, bbox, speed, centroid_history } = detection
        const [x1, y1, x2, y2] = bbox
        const isSelected = mode === 'target_vessels' && selectedTrackIds.has(track_id)

        // In all_detections mode, all boxes are blue. In target_vessels mode, selected are green, others red
        let strokeColor, fillColor, trailColor
        if (mode === 'all_detections') {
          strokeColor = '#00aaff'  // Blue for all detections
          fillColor = 'rgba(0, 170, 255, 0.7)'
          trailColor = 'rgba(0, 170, 255, 0.5)'
        } else {
          strokeColor = isSelected ? '#00ff00' : '#ff0000'  // Green if selected, red if not
          fillColor = isSelected ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)'
          trailColor = isSelected ? 'rgba(0, 255, 0, 0.5)' : 'rgba(255, 0, 0, 0.3)'
        }

        // Draw trajectory/tail using centroid history
        if (centroid_history && centroid_history.length > 1) {
          ctx.strokeStyle = trailColor
          ctx.lineWidth = 2
          ctx.beginPath()
          
          // Draw line connecting centroids (tail)
          for (let i = 0; i < centroid_history.length; i++) {
            const [cx, cy] = centroid_history[i]
            if (i === 0) {
              ctx.moveTo(cx, cy)
            } else {
              ctx.lineTo(cx, cy)
            }
          }
          ctx.stroke()
          
          // Draw dots for each centroid point
          ctx.fillStyle = trailColor
          centroid_history.forEach(([cx, cy]) => {
            ctx.beginPath()
            ctx.arc(cx, cy, 2, 0, 2 * Math.PI)
            ctx.fill()
          })
        }

        // Draw bounding box
        ctx.strokeStyle = strokeColor
        ctx.lineWidth = isSelected ? 3 : 2
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        // Draw label with speed
        const speedText = speed !== null && speed !== undefined ? ` | ${speed.toFixed(1)} px/s` : ''
        const label = `ID: ${track_id}${speedText}`
        ctx.font = '16px Arial'
        const textMetrics = ctx.measureText(label)
        const textWidth = textMetrics.width
        const textHeight = 20

        ctx.fillStyle = fillColor
        ctx.fillRect(x1, y1 - textHeight - 4, textWidth + 8, textHeight)

        // Draw label text
        ctx.fillStyle = '#ffffff'
        ctx.fillText(label, x1 + 4, y1 - 8)
      })
    }

    img.src = `data:image/jpeg;base64,${frame}`
    imageRef.current = img
  }, [frame, detections, selectedTrackIds, width, height, mode])

  const handleCanvasClick = (e) => {
    // Only allow clicking in target_vessels mode
    if (!onBboxClick || !canvasRef.current || detections.length === 0) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height

    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    // Find clicked bbox
    for (const detection of detections) {
      const [x1, y1, x2, y2] = detection.bbox
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        onBboxClick(detection.track_id)
        break
      }
    }
  }

  if (!frame) {
    return (
      <div className="video-display empty">
        <div className="empty-message">
          <p>No video loaded</p>
          <p className="hint">Select a video from the sidebar to begin</p>
        </div>
      </div>
    )
  }

  return (
    <div className="video-display">
      <canvas
        ref={canvasRef}
        className="video-canvas"
        onClick={handleCanvasClick}
        style={{ cursor: mode === 'target_vessels' ? 'pointer' : 'default' }}
      />
    </div>
  )
}

export default VideoDisplay

