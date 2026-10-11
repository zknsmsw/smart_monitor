// Tab switching
document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        if (btn.dataset.tab === 'faces') loadFaces();
    });
});

// Set default dates to today
const today = new Date().toISOString().split('T')[0];
document.getElementById('timeline-date').value = today;
document.getElementById('rec-date').value = today;

// Status polling
function checkStatus() {
    fetch('/api/status').then(r => r.json()).then(data => {
        const dot = document.getElementById('status-indicator');
        dot.classList.toggle('online', data.pipeline && data.pipeline.camera);
        dot.title = data.pipeline && data.pipeline.camera ? '摄像头在线' : '摄像头离线';
    }).catch(() => {});
}
setInterval(checkStatus, 5000);
checkStatus();

// Timeline
function loadTimeline() {
    const date = document.getElementById('timeline-date').value;
    fetch('/api/timeline?date=' + date).then(r => r.json()).then(data => {
        renderTimeline(data, date);
    });
    // Also load recordings for click-to-play
    fetch('/api/recordings?date=' + date).then(r => r.json()).then(data => {
        window._timelineRecordings = data;
    });
}

function renderTimeline(data, date) {
    const bar = document.getElementById('timeline-bar');
    bar.innerHTML = '';
    const totalMin = 24 * 60;

    // Hour labels
    for (let h = 0; h < 24; h += 3) {
        const label = document.createElement('span');
        label.className = 'hour-label';
        label.style.left = (h / 24 * 100) + '%';
        label.textContent = h + ':00';
        bar.appendChild(label);
    }

    // Motion blocks
    (data.motions || []).forEach(m => {
        const start = timeToMinutes(m.start_time);
        const end = m.end_time ? timeToMinutes(m.end_time) : start + 1;
        const block = document.createElement('div');
        block.className = 'motion-block';
        block.style.left = (start / totalMin * 100) + '%';
        block.style.width = Math.max(0.3, (end - start) / totalMin * 100) + '%';
        block.title = m.start_time.split('T')[1] || '';
        bar.appendChild(block);
    });

    // Face marks
    (data.faces || []).forEach(f => {
        const min = timeToMinutes(f.timestamp);
        const mark = document.createElement('div');
        mark.className = 'face-mark';
        mark.style.left = (min / totalMin * 100) + '%';
        mark.title = (f.label || '未知') + ' ' + (f.timestamp.split('T')[1] || '');
        bar.appendChild(mark);
    });
}

function timeToMinutes(isoStr) {
    const t = isoStr.split('T')[1];
    if (!t) return 0;
    const parts = t.split(':');
    return parseInt(parts[0]) * 60 + parseInt(parts[1]);
}

// Recordings
function loadRecordings() {
    const date = document.getElementById('rec-date').value;
    fetch('/api/recordings?date=' + date).then(r => r.json()).then(data => {
        const list = document.getElementById('rec-list');
        if (!data.length) { list.innerHTML = '<div class="no-data">当天无录像</div>'; return; }
        list.innerHTML = data.map(r => `
            <div class="rec-card" onclick="playRecording('${r.url}')">
                <span class="time">${(r.start_time || '').split('T')[1]?.substring(0,8) || ''}</span>
                <span class="meta">${Math.round(r.duration || 0)}秒 · ${Math.round((r.file_size || 0) / 1024)}KB</span>
                ${r.face_count ? '<span class="faces-badge">' + r.face_count + ' 人脸</span>' : ''}
            </div>
        `).join('');
    });
}

function playRecording(url) {
    const player = document.getElementById('player');
    player.src = url;
    player.play();
    // Switch to timeline tab to show player
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    document.querySelector('[data-tab="timeline"]').classList.add('active');
    document.getElementById('tab-timeline').classList.add('active');
}

// Faces
function loadFaces() {
    fetch('/api/faces').then(r => r.json()).then(data => {
        const grid = document.getElementById('face-grid');
        if (!data.length) { grid.innerHTML = '<div class="no-data">暂无人脸记录</div>'; return; }
        grid.innerHTML = data.map(g => `
            <div class="face-card" onclick="showFaceDetail(${g.id})">
                <img src="${g.crop_url || ''}" alt="face" onerror="this.style.display='none'">
                <input type="text" value="${g.label || ''}" placeholder="命名"
                    onclick="event.stopPropagation()"
                    onchange="renameFace(${g.id}, this.value)">
                <div class="count">出现 ${g.face_count} 次</div>
                <div class="count">${(g.last_seen || '').replace('T', ' ').substring(0, 16)}</div>
            </div>
        `).join('');
    });
}

function renameFace(groupId, label) {
    fetch('/api/faces/' + groupId, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({label})
    });
}

function showFaceDetail(groupId) {
    document.getElementById('face-grid').style.display = 'none';
    const detail = document.getElementById('face-detail');
    detail.style.display = 'block';
    fetch('/api/faces/' + groupId).then(r => r.json()).then(data => {
        const content = document.getElementById('face-detail-content');
        const label = data.label || '未命名 #' + data.id;
        content.innerHTML = `
            <h3 style="margin-bottom:8px">${label} (${data.face_count}次)</h3>
            <div class="face-events">
                ${(data.events || []).map(e => `
                    <div class="face-event">
                        <img src="${e.crop_url || ''}" alt="face" onerror="this.style.display='none'">
                        <div>${(e.timestamp || '').replace('T', ' ').substring(0, 19)}</div>
                        <div>置信度: ${(e.confidence || 0).toFixed(2)}</div>
                    </div>
                `).join('')}
            </div>
        `;
    });
}

function closeFaceDetail() {
    document.getElementById('face-detail').style.display = 'none';
    document.getElementById('face-grid').style.display = 'grid';
}

// Initial load
loadTimeline();
loadRecordings();
