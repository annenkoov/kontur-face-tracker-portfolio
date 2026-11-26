import QtQuick
import QtQuick.Layouts

GlassCard {
    id: panel
    property string observationText: ""
    RowLayout {
        anchors.fill: parent; spacing: 14
        Rectangle { Layout.preferredWidth: 48; Layout.preferredHeight: 48; radius: 14; color: "#292859"; clip: true
            Canvas {
                anchors.fill: parent; anchors.margins: 11
                onPaint: {
                    const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                    ctx.strokeStyle = "#a397ff"; ctx.lineWidth = 2.4; ctx.lineCap = "round"
                    ctx.beginPath(); ctx.moveTo(1, height * .52); ctx.lineTo(width * .25, height * .52)
                    ctx.lineTo(width * .39, height * .18); ctx.lineTo(width * .57, height * .82)
                    ctx.lineTo(width * .73, height * .42); ctx.lineTo(width - 1, height * .42); ctx.stroke()
                }
            }
        }
        Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: "#455174" }
        ColumnLayout { Layout.fillWidth: true; spacing: 4
            Text { text: "СРАВНЕНИЕ С ЛИЧНОЙ БАЗОЙ"; color: Theme.violet; font.pixelSize: 12; font.weight: Font.DemiBold }
            Text { text: panel.observationText; color: Theme.text; font.pixelSize: 13; wrapMode: Text.WordWrap; Layout.fillWidth: true; maximumLineCount: 2; elide: Text.ElideRight }
        }
    }
}
