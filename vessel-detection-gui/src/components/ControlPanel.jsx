import './ControlPanel.css'

function ControlPanel({ mode, onAllDetections, onTargetVessels, onTag, onTrack, onPause, isTracking, isPaused, selectedCount, taggedCount }) {
  return (
    <div className="control-panel">
      <h2>Controls</h2>
      
      {!mode && (
        <div className="mode-selection">
          <h3>Select Mode</h3>
          <button
            className="btn btn-all-detections"
            onClick={onAllDetections}
          >
            ALL DETECTIONS
          </button>
          <button
            className="btn btn-target-vessels"
            onClick={onTargetVessels}
          >
            TARGET VESSELS
          </button>
          <p className="mode-description">
            <strong>ALL DETECTIONS:</strong> Automatically track all detected vessels<br/>
            <strong>TARGET VESSELS:</strong> Select specific vessels to track
          </p>
        </div>
      )}

      {mode === 'all_detections' && (
        <div className="all-detections-mode">
          <div className="status">
            <div className="status-item">
              <span className="status-label">Mode:</span>
              <span className="status-value">All Detections</span>
            </div>
            <div className="status-item">
              <span className="status-label">State:</span>
              <span className="status-value">
                {isTracking ? 'Tracking All' : 'Paused'}
              </span>
            </div>
          </div>
          <div className="controls">
            <button
              className="btn btn-pause"
              onClick={onPause}
              disabled={!isTracking}
            >
              PAUSE
            </button>
          </div>
        </div>
      )}

      {mode === 'target_vessels' && (
        <>
          <div className="status">
            <div className="status-item">
              <span className="status-label">Mode:</span>
              <span className="status-value">Target Vessels</span>
            </div>
            <div className="status-item">
              <span className="status-label">Selected (Green):</span>
              <span className="status-value">{selectedCount} vessel(s)</span>
            </div>
            <div className="status-item">
              <span className="status-label">Tagged:</span>
              <span className="status-value">{taggedCount} vessel(s)</span>
            </div>
            <div className="status-item">
              <span className="status-label">State:</span>
              <span className="status-value">
                {isTracking ? 'Tracking' : isPaused ? 'Paused' : 'Ready'}
              </span>
            </div>
          </div>

          <div className="controls">
            <button
              className="btn btn-tag"
              onClick={onTag}
              disabled={selectedCount === 0}
            >
              TAG
            </button>
            
            <button
              className="btn btn-track"
              onClick={onTrack}
              disabled={taggedCount === 0 || isTracking}
              title={taggedCount === 0 ? "Click TAG first after selecting vessels" : ""}
            >
              TRACK
            </button>
            
            <button
              className="btn btn-pause"
              onClick={onPause}
              disabled={!isTracking}
            >
              PAUSE
            </button>
          </div>

          <div className="instructions">
            <h3>Instructions</h3>
            <ol>
              <li>Click on vessel bounding boxes to select them (they will turn green)</li>
              <li>Click <strong>TAG</strong> to confirm your selection</li>
              <li>Click <strong>TRACK</strong> to start tracking</li>
              <li>Click <strong>PAUSE</strong> to pause and re-tag if needed</li>
            </ol>
          </div>
        </>
      )}
    </div>
  )
}

export default ControlPanel

