import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

GlassCard {
    id: card
    property var backend: null

    RowLayout {
        anchors.fill: parent
        spacing: 16

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 8
            Text { text: "СЕЙЧАС"; color: Theme.violet; font.pixelSize: 13; font.weight: Font.DemiBold }

            RowLayout {
                Layout.fillWidth: true
                spacing: 14
                Canvas {
                    Layout.preferredWidth: 44; Layout.preferredHeight: 44
                    onPaint: {
                        const ctx = getContext("2d")
                        ctx.clearRect(0, 0, width, height)
                        ctx.strokeStyle = "#e9edff"; ctx.lineWidth = 2.2; ctx.lineCap = "round"
                        const s = 5, l = 11
                        ctx.beginPath()
                        ctx.moveTo(s, s + l); ctx.lineTo(s, s); ctx.lineTo(s + l, s)
                        ctx.moveTo(width - s - l, s); ctx.lineTo(width - s, s); ctx.lineTo(width - s, s + l)
                        ctx.moveTo(s, height - s - l); ctx.lineTo(s, height - s); ctx.lineTo(s + l, height - s)
                        ctx.moveTo(width - s - l, height - s); ctx.lineTo(width - s, height - s); ctx.lineTo(width - s, height - s - l)
                        ctx.stroke()
                        ctx.fillStyle = "#a397ff"
                        ctx.beginPath(); ctx.arc(width * .38, height * .48, 1.7, 0, Math.PI * 2); ctx.fill()
                        ctx.beginPath(); ctx.arc(width * .62, height * .48, 1.7, 0, Math.PI * 2); ctx.fill()
                    }
                }
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4
                    Text {
                        text: card.backend ? String(card.backend.stateTitle || "Ожидание лица") : "Ожидание лица"
                        color: Theme.text; font.pixelSize: 22; font.weight: Font.DemiBold
                        Layout.fillWidth: true; elide: Text.ElideRight
                    }
                    Text {
                        text: card.backend ? String(card.backend.stateDescription || "Начни с личной калибровки") : "Начни с личной калибровки"
                        color: Theme.muted; font.pixelSize: 14
                        Layout.fillWidth: true; elide: Text.ElideRight
                    }
                }
            }
        }

        Button {
            id: calibration
            Layout.preferredWidth: 142; Layout.preferredHeight: 44
            text: "Калибровка"
            onClicked: if (card.backend && typeof card.backend.calibrate === "function") card.backend.calibrate()
            background: Rectangle {
                radius: 13; clip: true
                color: calibration.down ? "#29365b" : calibration.hovered ? "#222e4d" : "#1a2340"
                border.width: calibration.activeFocus ? 2 : 1
                border.color: calibration.activeFocus ? "#f0f2ff" : "#606f9e"
            }
            contentItem: Text {
                text: calibration.text; color: Theme.text; font.pixelSize: 12; font.weight: Font.Medium
                leftPadding: 25; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
            }
            Item {
                width: 16; height: 16; anchors.left: parent.left; anchors.leftMargin: 17; anchors.verticalCenter: parent.verticalCenter
                Canvas {
                    anchors.fill: parent
                    onPaint: {
                        const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                        ctx.strokeStyle = Theme.text; ctx.lineWidth = 1.5
                        ctx.beginPath(); ctx.arc(8, 8, 5.5, 0, Math.PI * 2); ctx.stroke()
                        ctx.beginPath(); ctx.moveTo(8, 0); ctx.lineTo(8, 4); ctx.moveTo(8, 12); ctx.lineTo(8, 16)
                        ctx.moveTo(0, 8); ctx.lineTo(4, 8); ctx.moveTo(12, 8); ctx.lineTo(16, 8); ctx.stroke()
                        ctx.beginPath(); ctx.arc(8, 8, 1, 0, Math.PI * 2); ctx.stroke()
                    }
                }
            }
        }
    }
}
