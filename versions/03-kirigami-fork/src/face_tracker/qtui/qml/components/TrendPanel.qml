import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

GlassCard {
    id: panel
    property var scoreHistory: []
    property bool hasScore: false
    property bool collectingBaseline: false
    property real scoreValue: 0

    ColumnLayout {
        anchors.fill: parent
        spacing: 8
        RowLayout {
            Layout.fillWidth: true
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 3
                Text { text: "ДИНАМИКА СЕССИИ"; color: Theme.violet; font.pixelSize: 15; font.weight: Font.DemiBold }
                Text {
                    objectName: "trendScaleHint"
                    text: "0 — мало изменений\n100 — сильное отклонение от личной базы"
                    color: Theme.muted
                    font.pixelSize: 13
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    Layout.preferredHeight: implicitHeight
                }
            }
            Text {
                objectName: "trendValueLabel"
                text: panel.hasScore ? Math.round(panel.scoreValue) + " / 100" : (panel.collectingBaseline ? "Собираем базу" : "Нет данных")
                color: Theme.text; font.pixelSize: 14; font.weight: Font.DemiBold
            }
            ComboBox {
                id: periodCombo
                model: ["Последние 60 секунд", "Последние 5 минут"]
                currentIndex: 0
                implicitWidth: 168
                implicitHeight: 34
                onCurrentIndexChanged: chart.requestPaint()
                background: Rectangle { radius: 10; color: "#151d34"; border.color: "#59699a" }
                contentItem: Text { leftPadding: 12; text: periodCombo.displayText; color: Theme.text; verticalAlignment: Text.AlignVCenter; font.pixelSize: 13; elide: Text.ElideRight }
            }
        }
        Canvas {
            id: chart; objectName: "trendCanvas"; Layout.fillWidth: true; Layout.fillHeight: true
            onPaint: {
                const ctx = getContext("2d"), w = width, h = height
                ctx.clearRect(0, 0, w, h)
                const left = 38, right = w - 10, top = 12, bottom = h - 26
                const plotW = right - left, plotH = bottom - top
                const bg = ctx.createLinearGradient(0, top, 0, bottom)
                bg.addColorStop(0, "rgba(55, 55, 108, 0.70)"); bg.addColorStop(1, "rgba(16, 43, 60, 0.72)")
                const radius = 14
                ctx.beginPath()
                ctx.moveTo(left + radius, top); ctx.lineTo(right - radius, top)
                ctx.quadraticCurveTo(right, top, right, top + radius)
                ctx.lineTo(right, bottom - radius); ctx.quadraticCurveTo(right, bottom, right - radius, bottom)
                ctx.lineTo(left + radius, bottom); ctx.quadraticCurveTo(left, bottom, left, bottom - radius)
                ctx.lineTo(left, top + radius); ctx.quadraticCurveTo(left, top, left + radius, top)
                ctx.closePath(); ctx.fillStyle = bg; ctx.fill(); ctx.save(); ctx.clip()
                ctx.font = "11px Segoe UI"; ctx.textBaseline = "middle"
                for (let i = 0; i <= 4; ++i) {
                    const y = top + plotH * i / 4
                    ctx.strokeStyle = "#435071"; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke()
                }
                ctx.setLineDash([])
                const periodSeconds = periodCombo.currentIndex === 1 ? 300 : 60
                const raw = panel.scoreHistory || []
                const entries = raw.map((entry, i) => {
                    // Явный null сохраняем как точку-разрыв, чтобы график не соединял данные через пропуск.
                    if (entry === null || entry === undefined) return { gap: true, timestamp: NaN, value: null, fallbackIndex: i }
                    if (typeof entry === "object") {
                        const value = entry.value === null || entry.value === undefined ? null : Number(entry.value)
                        const timestamp = entry.timestamp === null || entry.timestamp === undefined ? NaN : Number(entry.timestamp)
                        return { gap: false, timestamp: timestamp, value: value, fallbackIndex: i }
                    }
                    return { gap: false, timestamp: NaN, value: Number(entry), fallbackIndex: i }
                })
                const timed = entries.filter(entry => !entry.gap && Number.isFinite(entry.timestamp))
                const latestTime = timed.length ? Math.max(...timed.map(entry => entry.timestamp)) : entries.length - 1
                const windowStart = latestTime - periodSeconds
                const visibleEntries = entries.filter(entry => {
                    const t = Number.isFinite(entry.timestamp) ? entry.timestamp : latestTime - (entries.length - 1 - entry.fallbackIndex)
                    entry.plotTime = t
                    return t >= windowStart
                })
                const valid = visibleEntries.filter(entry => !entry.gap && entry.value !== null && Number.isFinite(entry.value))
                if (valid.length > 0) {
                    const pts = visibleEntries.map(entry => entry.gap || entry.value === null || !Number.isFinite(entry.value) ? null : ({
                        x: left + Math.max(0, Math.min(1, (entry.plotTime - windowStart) / periodSeconds)) * plotW,
                        y: bottom - Math.max(0, Math.min(100, entry.value)) / 100 * plotH
                    }))
                    let run = []
                    const traceCurve = function(points, moveToStart) {
                        if (moveToStart) ctx.moveTo(points[0].x, points[0].y)
                        for (let i = 1; i < points.length; i++) {
                            const prev = points[i - 1], cur = points[i], mx = (prev.x + cur.x) / 2
                            ctx.quadraticCurveTo(prev.x, prev.y, mx, (prev.y + cur.y) / 2)
                        }
                        if (points.length > 1)
                            ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y)
                    }
                    const strokeRun = function(points) {
                        if (points.length < 1) return
                        ctx.beginPath(); traceCurve(points, true)
                        ctx.stroke()
                    }
                    const line = ctx.createLinearGradient(left, 0, right, 0); line.addColorStop(0, "#7479ff"); line.addColorStop(1, "#42dfd4")
                    ctx.strokeStyle = line; ctx.lineWidth = 3
                    for (let i = 0; i <= pts.length; i++) {
                        if (i < pts.length && pts[i] !== null) { run.push(pts[i]); continue }
                        if (run.length) {
                            strokeRun(run)
                            if (run.length === 1) {
                                ctx.fillStyle = "#9188ff"
                                ctx.beginPath(); ctx.arc(run[0].x, run[0].y, 4, 0, Math.PI * 2); ctx.fill()
                            } else if (run.length > 1) {
                                ctx.beginPath(); ctx.moveTo(run[0].x, bottom); ctx.lineTo(run[0].x, run[0].y)
                                traceCurve(run, false)
                                ctx.lineTo(run[run.length - 1].x, bottom); ctx.closePath()
                                const area = ctx.createLinearGradient(0, top, 0, bottom)
                                area.addColorStop(0, "rgba(130, 118, 255, 0.24)"); area.addColorStop(1, "rgba(67, 223, 212, 0.025)")
                                ctx.fillStyle = area; ctx.fill()
                            }
                            run = []
                        }
                    }
                } else {
                    ctx.textAlign = "center"; ctx.fillStyle = "#a9b3d0"; ctx.font = "15px Segoe UI"
                    const emptyText = panel.collectingBaseline
                        ? "Собираем личную базу — оставайтесь перед камерой"
                        : "Создайте личную базу, затем начните сессию"
                    ctx.fillText(emptyText, left + plotW / 2, top + plotH / 2)
                }
                ctx.restore()
                ctx.textBaseline = "middle"; ctx.textAlign = "right"; ctx.fillStyle = "#aab4d0"; ctx.font = "13px Segoe UI"
                for (let i = 0; i <= 4; ++i) {
                    const y = top + plotH * i / 4
                    ctx.fillText(String(100 - i * 25), left - 8, y)
                }
                ctx.textBaseline = "alphabetic"; ctx.textAlign = "left"; ctx.fillStyle = "#a9b3d0"
                ctx.fillText(periodCombo.currentIndex === 1 ? "−5 мин" : "−60 с", left, h - 9); ctx.textAlign = "right"; ctx.fillText("Сейчас", right, h - 9)
            }
            Connections { target: panel; function onScoreHistoryChanged() { chart.requestPaint() } }
            onWidthChanged: requestPaint(); onHeightChanged: requestPaint()
        }
    }
}
