const data = window.MODEL_COMPARISON_DATA;

const nf1 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });

function formatNumber(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return digits === 0 ? `${Math.round(value)}` : new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: digits
    }).format(value);
}

function formatResolution(value) {
    if (value === null || value === undefined) return '—';
    return `${value} px`;
}

function formatMetric(value, digits = 1, suffix = '') {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return `${formatNumber(value, digits)}${suffix}`;
}

function makeLink(url, label) {
    if (!url) return '';
    return `<a class="source-link" href="${url}" target="_blank" rel="noreferrer">${label}</a>`;
}

function flattenModels(models) {
    const rows = [];

    for (const model of models) {
        if (Array.isArray(model.variants) && model.variants.length > 0) {
            for (const variant of model.variants) {
                rows.push({
                    ...model,
                    ...variant,
                    parent_name: model.model_name,
                    family: model.family,
                    display_name: variant.model_name,
                    variant: true
                });
            }
            continue;
        }

        rows.push({
            ...model,
            display_name: model.model_name,
            variant: false
        });
    }

    return rows;
}

function getPrimaryAp(row) {
    return row?.coco?.ap ?? row?.ap ?? null;
}

function getPrimaryAp50(row) {
    return row?.coco?.ap50 ?? row?.ap50 ?? null;
}

function getLatency(row) {
    return row?.latency_ms ?? row?.latency_ms_t4_tensorrt_fp16 ?? row?.latency_ms_t4 ?? null;
}

function getFps(row) {
    return row?.fps ?? row?.fps_t4 ?? null;
}

function getParams(row) {
    return row?.params_millions ?? row?.model_metadata?.parameters_millions ?? null;
}

function getGflops(row) {
    return row?.gflops ?? row?.model_metadata?.gflops ?? null;
}

function getSortScore(row) {
    const ap = getPrimaryAp(row);
    return ap === null ? -Infinity : ap;
}

function compareNullableDesc(a, b) {
    const aVal = a === null || a === undefined ? -Infinity : a;
    const bVal = b === null || b === undefined ? -Infinity : b;
    return bVal - aVal;
}

function compareNullableAsc(a, b) {
    const aVal = a === null || a === undefined ? Infinity : a;
    const bVal = b === null || b === undefined ? Infinity : b;
    return aVal - bVal;
}

function metricBadge(label, value) {
    return `<span class="metric-chip"><span class="small-label">${label}</span><span class="metric-value">${value}</span></span>`;
}

function renderHeroMeta(meta) {
    const heroMeta = document.getElementById('heroMeta');
    heroMeta.innerHTML = [
        `<span class="meta-chip">${meta.benchmark_dataset}</span>`,
        `<span class="meta-chip">${meta.created}</span>`,
        `<span class="meta-chip">${data.models.length} model entries</span>`,
        `<span class="meta-chip">${data.models.reduce((count, model) => count + (model.variants ? model.variants.length : 1), 0)} comparison rows</span>`
    ].join('');
}

function renderSnapshot(rows) {
    const snapshotGrid = document.getElementById('snapshotGrid');
    const bestAp = rows.filter((row) => getPrimaryAp(row) !== null).sort((a, b) => compareNullableDesc(getPrimaryAp(a), getPrimaryAp(b)))[0];
    const fastest = rows.filter((row) => getLatency(row) !== null).sort((a, b) => compareNullableAsc(getLatency(a), getLatency(b)))[0];
    const fastestFps = rows.filter((row) => getFps(row) !== null).sort((a, b) => compareNullableDesc(getFps(a), getFps(b)))[0];
    const largest = rows.filter((row) => getParams(row) !== null).sort((a, b) => compareNullableDesc(getParams(a), getParams(b)))[0];
    const newest = rows.slice().sort((a, b) => (b.year_released ?? 0) - (a.year_released ?? 0))[0];

    const cards = [
        [
            'Best AP',
            bestAp ? `${formatNumber(getPrimaryAp(bestAp), 1)} by ${bestAp.display_name}` : '—',
            'Primary COCO metric'
        ],
        [
            'Fastest latency',
            fastest ? `${formatMetric(getLatency(fastest), 2, ' ms')}` : '—',
            fastest ? fastest.display_name : 'Latency unavailable'
        ],
        [
            'Top throughput',
            fastestFps ? `${formatMetric(getFps(fastestFps), 0, ' FPS')}` : '—',
            fastestFps ? fastestFps.display_name : 'FPS unavailable'
        ],
        [
            'Largest footprint',
            largest ? `${formatMetric(getParams(largest), 1, ' M')}` : '—',
            largest ? largest.display_name : 'Parameters unavailable'
        ],
        [
            'Newest release',
            newest ? `${newest.year_released}` : '—',
            newest ? newest.display_name : ''
        ]
    ];

    snapshotGrid.innerHTML = cards.map(([label, value, note]) => `
        <article class="snapshot-card">
            <span class="snapshot-label">${label}</span>
            <span class="snapshot-value">${value}</span>
            <span class="snapshot-note">${note}</span>
        </article>
    `).join('');
}

function renderNotes(notes) {
    const notesGrid = document.getElementById('notesGrid');
    notesGrid.innerHTML = notes.map((note, index) => `
        <article class="note-card">
            <span class="small-label">Note ${index + 1}</span>
            <p>${note}</p>
        </article>
    `).join('');
}

function renderBarChart(containerId, rows, options) {
    const container = document.getElementById(containerId);
    const metricValues = rows.map(options.getValue).filter((value) => value !== null && value !== undefined);

    if (metricValues.length === 0) {
        container.innerHTML = '<div class="empty-state">No published values for this metric.</div>';
        return;
    }

    const max = Math.max(...metricValues);
    const min = Math.min(...metricValues);
    const range = max - min || 1;

    container.innerHTML = rows.map((row) => {
        const value = options.getValue(row);
        if (value === null || value === undefined) {
            return `
                <div class="bar-row">
                    <div class="bar-label">
                        <span class="bar-name">${row.display_name}</span>
                        <span class="bar-sub">${row.family}</span>
                    </div>
                    <div class="bar-track"><div class="bar-fill" style="width: 0%"></div></div>
                    <div class="bar-value">—</div>
                </div>
            `;
        }

        const width = options.lowerIsBetter
            ? ((max - value) / range) * 100
            : (value / max) * 100;
        const safeWidth = Math.max(6, Math.min(100, width));
        const fillClass = options.fillClass || '';
        const displayValue = options.format(value);

        return `
            <div class="bar-row">
                <div class="bar-label">
                    <span class="bar-name">${row.display_name}</span>
                    <span class="bar-sub">${row.family}</span>
                </div>
                <div class="bar-track"><div class="bar-fill ${fillClass}" style="width: ${safeWidth}%"></div></div>
                <div class="bar-value">${displayValue}</div>
            </div>
        `;
    }).join('');
}

function renderModelCards(rows) {
    const cards = document.getElementById('modelCards');
    cards.innerHTML = rows.map((row) => {
        const ap = getPrimaryAp(row);
        const ap50 = getPrimaryAp50(row);
        const latency = getLatency(row);
        const fps = getFps(row);
        const params = getParams(row);
        const gflops = getGflops(row);
        const inputResolution = row.input_resolution ?? null;
        const sourceNotes = Array.isArray(row.source_notes) ? row.source_notes : [];
        const extraTitle = row.variant ? row.parent_name : row.architecture_type;

        return `
            <article class="model-card">
                <div class="model-topline">
                    <div>
                        <span class="model-family">${row.family}</span>
                        <h3>${row.display_name}</h3>
                        <p class="card-muted">${extraTitle}</p>
                    </div>
                    <div class="model-score">AP ${ap === null ? '—' : formatNumber(ap, 1)}</div>
                </div>
                <div class="model-tag-row">
                    <span class="table-pill">${row.detection_type}</span>
                    <span class="table-pill">${row.backbone}</span>
                    <span class="table-pill">${row.year_released}</span>
                </div>
                <div class="metric-grid">
                    <div class="model-metric">
                        <span class="metric-label">COCO AP</span>
                        <span class="metric-value ${ap === null ? 'missing' : ''}">${ap === null ? '—' : formatNumber(ap, 1)}</span>
                        <span class="metric-note">Primary benchmark</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">AP50</span>
                        <span class="metric-value ${ap50 === null ? 'missing' : ''}">${ap50 === null ? '—' : formatNumber(ap50, 1)}</span>
                        <span class="metric-note">COCO AP at 0.50 IoU</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">Latency</span>
                        <span class="metric-value ${latency === null ? 'missing' : ''}">${latency === null ? '—' : formatMetric(latency, 2, ' ms')}</span>
                        <span class="metric-note">T4 / TensorRT FP16 where published</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">FPS</span>
                        <span class="metric-value ${fps === null ? 'missing' : ''}">${fps === null ? '—' : formatMetric(fps, 0)}</span>
                        <span class="metric-note">Throughput on T4 where available</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">Params</span>
                        <span class="metric-value ${params === null ? 'missing' : ''}">${params === null ? '—' : formatMetric(params, 1, ' M')}</span>
                        <span class="metric-note">Trainable parameters</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">GFLOPs</span>
                        <span class="metric-value ${gflops === null ? 'missing' : ''}">${gflops === null ? '—' : formatMetric(gflops, 1)}</span>
                        <span class="metric-note">Reported compute</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">Input</span>
                        <span class="metric-value ${inputResolution === null ? 'missing' : ''}">${inputResolution === null ? '—' : formatResolution(inputResolution)}</span>
                        <span class="metric-note">Benchmark resolution</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">License</span>
                        <span class="metric-value">${row.license}</span>
                        <span class="metric-note">Usage / redistribution</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">Family</span>
                        <span class="metric-value">${row.family}</span>
                        <span class="metric-note">Comparison group</span>
                    </div>
                </div>
                <div class="sources">
                    ${makeLink(row.paper_url, 'Paper')}
                    ${makeLink(row.code_url, 'Code')}
                    ${row.performance_source ? makeLink(row.performance_source, 'Benchmarks') : ''}
                    ${sourceNotes.length ? `<span class="table-pill">${sourceNotes.join(' · ')}</span>` : ''}
                </div>
            </article>
        `;
    }).join('');
}

function renderComparisonTable(rows) {
    const body = document.getElementById('comparisonTableBody');
    body.innerHTML = rows.map((row) => {
        const ap = getPrimaryAp(row);
        const ap50 = getPrimaryAp50(row);
        const latency = getLatency(row);
        const fps = getFps(row);
        const params = getParams(row);
        const gflops = getGflops(row);
        const inputResolution = row.input_resolution ?? '—';

        return `
            <tr>
                <td>
                    <div class="row-top">
                        <span class="model-badge">${row.family}</span>
                        <strong>${row.display_name}</strong>
                    </div>
                    <div class="cell-small">${row.detection_type}</div>
                </td>
                <td>${row.family}</td>
                <td>${ap === null ? '—' : formatNumber(ap, 1)}</td>
                <td>${ap50 === null ? '—' : formatNumber(ap50, 1)}</td>
                <td>${latency === null ? '—' : formatMetric(latency, 2)}</td>
                <td>${fps === null ? '—' : formatMetric(fps, 0)}</td>
                <td>${params === null ? '—' : formatMetric(params, 1)}</td>
                <td>${gflops === null ? '—' : formatMetric(gflops, 1)}</td>
                <td>${inputResolution === '—' ? '—' : `${inputResolution}`}</td>
                <td>${row.license}</td>
            </tr>
        `;
    }).join('');
}

function buildPage() {
    const rows = flattenModels(data.models).sort((a, b) => compareNullableDesc(getSortScore(a), getSortScore(b)));

    renderHeroMeta(data.metadata);
    renderSnapshot(rows);
    renderNotes(data.metadata.notes);

    renderBarChart('apChart', rows.filter((row) => getPrimaryAp(row) !== null), {
        getValue: getPrimaryAp,
        format: (value) => formatMetric(value, 1),
        fillClass: '',
        lowerIsBetter: false
    });

    renderBarChart('latencyChart', rows.filter((row) => getLatency(row) !== null), {
        getValue: getLatency,
        format: (value) => formatMetric(value, 2, ' ms'),
        fillClass: 'alt',
        lowerIsBetter: true
    });

    renderBarChart('fpsChart', rows.filter((row) => getFps(row) !== null), {
        getValue: getFps,
        format: (value) => formatMetric(value, 0, ' FPS'),
        fillClass: 'danger',
        lowerIsBetter: false
    });

    renderModelCards(rows);
    renderComparisonTable(rows);
}

buildPage();
