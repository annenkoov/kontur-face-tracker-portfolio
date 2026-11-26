pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts

GlassCard {
    id: card
    property var backend: null
    property bool hasScore: backend ? Boolean(backend.hasScore) : false
    property real scoreValue: backend ? Number(backend.scoreValue || 0) : 0
    property string expression: backend ? String(backend.expressionText || "Ожидание лица") : "Ожидание лица"
    property string expressionAgreement: backend ? String(backend.expressionAgreementText || "—") : "—"
    property string quality: backend ? String(backend.qualityText || "Ожидание сигнала") : "Ожидание сигнала"
    property int qualityLevel: quality === "Высокое" ? 5 : quality === "Среднее" ? 3 : quality === "Низкое" ? 1 : 0

    ColumnLayout {
        anchors.fill: parent
        spacing: 10
        Text { text: "ИЗМЕНЕНИЕ МИМИКИ"; color: Theme.violet; font.pixelSize: 13; font.weight: Font.DemiBold }

        RowLayout {
            Layout.fillWidth: true; Layout.fillHeight: true; spacing: 22
            Item {
                Layout.preferredWidth: 194; Layout.fillHeight: true
                Canvas {
                    id: gauge
                    anchors.centerIn: parent
                    width: Math.min(parent.width, parent.height, 184); height: width
                    onPaint: {
                        const ctx = getContext("2d")
                        ctx.clearRect(0, 0, width, height)
                        const r = width * .40, cx = width / 2, cy = height / 2
                        ctx.lineWidth = 12; ctx.lineCap = "round"
                        ctx.strokeStyle = "#293657"
                        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke()
                        if (card.hasScore) {
                            const arc = ctx.createLinearGradient(0, 0, width, height)
                            arc.addColorStop(0, "#8d81ff"); arc.addColorStop(1, "#42dfd4")
                            ctx.strokeStyle = arc
                            ctx.beginPath()
                            ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.max(0, Math.min(100, card.scoreValue)) / 100)
                            ctx.stroke()
                        }
                    }
                    Connections {
                        target: card
                        function onScoreValueChanged() { gauge.requestPaint() }
                        function onHasScoreChanged() { gauge.requestPaint() }
                    }
                }
                Column {
                    anchors.centerIn: parent
                    spacing: 1
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: card.hasScore ? Math.round(card.scoreValue) : "—"
                        color: Theme.text; font.pixelSize: 44; font.weight: Font.DemiBold
                    }
                    Text { anchors.horizontalCenter: parent.horizontalCenter; text: "из 100"; color: Theme.muted; font.pixelSize: 12 }
                }
            }

            Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: "#3b4969" }

            ColumnLayout {
                Layout.fillWidth: true; spacing: 12
                RowLayout {
                    Layout.fillWidth: true; spacing: 12
                    Canvas {
                        Layout.preferredWidth: 24; Layout.preferredHeight: 28
                        onPaint: {
                            const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                            ctx.strokeStyle = Theme.violet; ctx.lineWidth = 1.8
                            ctx.beginPath(); ctx.arc(12, 7, 4, 0, Math.PI * 2); ctx.stroke()
                            ctx.beginPath(); ctx.moveTo(4, 25); ctx.lineTo(5, 20); ctx.quadraticCurveTo(12, 13, 19, 20); ctx.lineTo(20, 25); ctx.stroke()
                        }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true; spacing: 3
                        Text { text: "Выражение лица"; color: Theme.muted; font.pixelSize: 13 }
                        Text {
                            text: card.expression
                            color: Theme.text; font.pixelSize: 16; font.weight: Font.DemiBold
                            Layout.fillWidth: true; wrapMode: Text.WordWrap
                            maximumLineCount: 2; elide: Text.ElideRight
                        }
                    }
                }
                Rectangle { Layout.fillWidth: true; height: 1; color: "#303d5d" }
                RowLayout {
                    Layout.fillWidth: true; spacing: 12
                    Canvas {
                        Layout.preferredWidth: 24; Layout.preferredHeight: 24
                        onPaint: {
                            const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                            ctx.strokeStyle = Theme.cyan; ctx.lineWidth = 1.8; ctx.lineJoin = "round"
                            ctx.strokeRect(2, 5, 13, 13)
                            ctx.beginPath(); ctx.moveTo(15, 9); ctx.lineTo(22, 5); ctx.lineTo(22, 18); ctx.lineTo(15, 14); ctx.closePath(); ctx.stroke()
                        }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true; spacing: 3
                        Text { text: "Качество кадра"; color: Theme.muted; font.pixelSize: 13 }
                        Text { text: card.quality; color: Theme.text; font.pixelSize: 16; font.weight: Font.DemiBold; Layout.fillWidth: true; elide: Text.ElideRight }
                    }
                    Row { spacing: 3; Layout.alignment: Qt.AlignBottom
                        Repeater {
                            model: 5
                            delegate: Rectangle {
                                required property int index
                                width: 7; height: 6 + index * 2; radius: 2; anchors.bottom: parent.bottom
                                color: index < card.qualityLevel ? Theme.cyan : "#35415e"
                            }
                        }
                    }
                }
                Text {
                    text: "Согласие сигнала · " + card.expressionAgreement
                    color: Theme.muted; font.pixelSize: 12; Layout.fillWidth: true; elide: Text.ElideRight
                }
            }
        }
    }
}
