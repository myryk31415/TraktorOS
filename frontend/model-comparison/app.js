function fmt(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return Number(value).toFixed(digits);
}

function fmtInt(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return `${Math.round(Number(value))}`;
}

function modelRows(payload) {
    const rows = Array.isArray(payload?.models) ? payload.models.slice() : [];
    rows.sort((a, b) => (Number(b.f1 || 0) - Number(a.f1 || 0)));
    return rows;
}

function renderHeroMeta(payload) {
    const meta = document.getElementById('heroMeta');
    const settings = payload.settings || {};
    const summary = payload.summary || {};

    meta.innerHTML = [
        `<span class="meta-chip">Device: ${settings.device || '—'}</span>`,
        `<span class="meta-chip">Images: ${fmtInt(settings.max_images)}</span>`,
        `<span class="meta-chip">Conf: ${fmt(settings.confidence_threshold, 2)}</span>`,
        `<span class="meta-chip">IoU: ${fmt(settings.iou_threshold, 2)}</span>`,
        `<span class="meta-chip">Total time: ${fmt(summary.overall_elapsed_seconds, 2)} s</span>`
    ].join('');
}

function renderSnapshot(payload, rows) {
    const grid = document.getElementById('snapshotGrid');
    const bestF1 = rows[0];

    const bestPrecision = rows.slice().sort((a, b) => Number(b.precision || 0) - Number(a.precision || 0))[0];
    const bestRecall = rows.slice().sort((a, b) => Number(b.recall || 0) - Number(a.recall || 0))[0];
    const fastest = rows.slice().sort((a, b) => Number(b.throughput_images_per_second || 0) - Number(a.throughput_images_per_second || 0))[0];

    const cards = [
        ['Best F1', bestF1 ? `${fmt(bestF1.f1)} (${bestF1.model})` : '—', 'Main quality indicator'],
        ['Best precision', bestPrecision ? `${fmt(bestPrecision.precision)} (${bestPrecision.model})` : '—', 'Fewest false alarms'],
        ['Best recall', bestRecall ? `${fmt(bestRecall.recall)} (${bestRecall.model})` : '—', 'Most detections found'],
        ['Fastest', fastest ? `${fmt(fastest.throughput_images_per_second, 2)} img/s (${fastest.model})` : '—', 'Runtime throughput'],
    ];

    grid.innerHTML = cards.map(([label, value, note]) => `
        <article class="snapshot-card">
            <span class="snapshot-label">${label}</span>
            <span class="snapshot-value">${value}</span>
            <span class="snapshot-note">${note}</span>
        </article>
    `).join('');
}

function renderF1Chart(rows) {
    const chart = document.getElementById('f1Chart');
    const f1Values = rows.map((row) => Number(row.f1 || 0));
    const maxF1 = Math.max(...f1Values, 1e-6);

    chart.innerHTML = rows.map((row) => {
        const f1 = Number(row.f1 || 0);
        const f1Width = Math.max(6, (f1 / maxF1) * 100);
        return `
            <div class="bar-row">
                <div class="bar-label">
                    <span class="bar-name">${row.model}</span>
                    <span class="bar-sub">Precision ${fmt(row.precision)} · Recall ${fmt(row.recall)}</span>
                </div>
                <div class="bar-track"><div class="bar-fill" style="width: ${f1Width}%"></div></div>
                <div class="bar-value">${fmt(f1)}</div>
            </div>
        `;
    }).join('');
}

function renderThroughputChart(rows) {
    const chart = document.getElementById('imgPerSecChart');
    const ipsValues = rows.map((row) => Number(row.throughput_images_per_second || 0));
    const maxIps = Math.max(...ipsValues, 1e-6);

    chart.innerHTML = rows.map((row) => {
        const imgPerSec = Number(row.throughput_images_per_second || 0);
        const ipsWidth = Math.max(6, (imgPerSec / maxIps) * 100);
        return `
            <div class="bar-row">
                <div class="bar-label">
                    <span class="bar-name">${row.model}</span>
                    <span class="bar-sub">Elapsed ${fmt(row.elapsed_seconds, 2)} s</span>
                </div>
                <div class="bar-track"><div class="bar-fill alt" style="width: ${ipsWidth}%"></div></div>
                <div class="bar-value">${fmt(imgPerSec, 2)}</div>
            </div>
        `;
    }).join('');
}

function renderTable(rows) {
    const body = document.getElementById('comparisonTableBody');
    body.innerHTML = rows.map((row) => `
        <tr>
            <td><strong>${row.model}</strong></td>
            <td>${fmt(row.f1)}</td>
            <td>${fmt(row.precision)}</td>
            <td>${fmt(row.recall)}</td>
            <td>${fmtInt(row.tp)}</td>
            <td>${fmtInt(row.fp)}</td>
            <td>${fmtInt(row.fn)}</td>
            <td>${fmt(row.mean_iou)}</td>
            <td>${fmt(row.throughput_images_per_second, 2)}</td>
            <td>${fmt(row.elapsed_seconds, 2)}</td>
        </tr>
    `).join('');
}

async function loadAndRender() {
    const paths = [
        '../../results/model_eval_results.json',
        '/results/model_eval_results.json',
        'model_eval_results.json'
    ];

    let payload = null;
    let lastError = null;

    for (const path of paths) {
        try {
            const response = await fetch(path, { cache: 'no-store' });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            payload = await response.json();
            break;
        } catch (error) {
            lastError = error;
        }
    }

    if (!payload) {
        const body = document.getElementById('comparisonTableBody');
        const isFileProtocol = window.location.protocol === 'file:';
        const hint = isFileProtocol
            ? 'Open this page through a local HTTP server (not file://), e.g. from repo root: python3 -m http.server 8000 and visit http://localhost:8000/frontend/model-comparison/index.html'
            : 'Verify that results/model_eval_results.json is reachable from the current host.';
        body.innerHTML = `<tr><td colspan="10">Failed to load results JSON. ${String(lastError || '')}<br>${hint}</td></tr>`;
        return;
    }

    const rows = modelRows(payload);
    renderHeroMeta(payload);
    renderSnapshot(payload, rows);
    renderF1Chart(rows);
    renderThroughputChart(rows);
    renderTable(rows);
}

loadAndRender();
