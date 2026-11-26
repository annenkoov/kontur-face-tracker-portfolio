import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

GlassCard {
    id: bar
    property var backend: null

    RowLayout {
        anchors.fill: parent
        spacing: 12
        Button {
            id: createButton
            scale: down ? 0.98 : 1.0
            Behavior on scale { NumberAnimation { duration: createButton.down ? 90 : 140; easing.type: Easing.OutCubic } }
            Layout.fillWidth: true; Layout.fillHeight: true; text: "Создать базу"
            onClicked: if (bar.backend && typeof bar.backend.createBaseline === "function") bar.backend.createBaseline()
            background: Rectangle { radius: 15; color: createButton.down ? "#26345b" : createButton.hovered ? "#202c49" : "#151e36"; border.color: createButton.activeFocus ? "#f0f2ff" : "#6275aa"; border.width: createButton.activeFocus ? 2 : 1; clip: true }
            contentItem: RowLayout {
                anchors.centerIn: parent; spacing: 10
                Canvas { Layout.preferredWidth: 22; Layout.preferredHeight: 24
                    onPaint: {
                        const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                        ctx.strokeStyle = Theme.text; ctx.lineWidth = 1.7
                        ctx.beginPath(); ctx.ellipse(3, 2, 16, 6); ctx.stroke()
                        ctx.beginPath(); ctx.moveTo(3, 2); ctx.lineTo(3, 20); ctx.ellipse(3, 14, 16, 6); ctx.stroke()
                        ctx.beginPath(); ctx.moveTo(19, 2); ctx.lineTo(19, 20); ctx.stroke()
                    }
                }
                Text { text: createButton.text; color: Theme.text; font.pixelSize: 16; Layout.alignment: Qt.AlignVCenter }
            }
        }
        Button {
            id: startButton
            scale: down ? 0.98 : 1.0
            Behavior on scale { NumberAnimation { duration: startButton.down ? 90 : 140; easing.type: Easing.OutCubic } }
            Layout.fillWidth: true; Layout.fillHeight: true; text: "Начать сессию"
            onClicked: if (bar.backend && typeof bar.backend.startSession === "function") bar.backend.startSession()
            background: Rectangle {
                radius: 15
                gradient: Gradient {
                    GradientStop { position: 0; color: "#665fe9" }
                    GradientStop { position: 1; color: "#8374ff" }
                }
                border.color: startButton.activeFocus ? "#f0f2ff" : startButton.hovered ? "#c1baff" : "#aaa2ff"
                border.width: startButton.activeFocus ? 2 : 1
                clip: true
            }
            contentItem: RowLayout {
                anchors.centerIn: parent; spacing: 10
                Canvas { Layout.preferredWidth: 22; Layout.preferredHeight: 24
                    onPaint: {
                        const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.lineJoin = "round"
                        ctx.beginPath(); ctx.moveTo(4, 2); ctx.lineTo(20, 12); ctx.lineTo(4, 22); ctx.closePath(); ctx.stroke()
                    }
                }
                Text { text: startButton.text; color: "white"; font.pixelSize: 17; font.weight: Font.DemiBold; Layout.alignment: Qt.AlignVCenter }
            }
        }
        Button {
            id: stopButton
            scale: down ? 0.98 : 1.0
            Behavior on scale { NumberAnimation { duration: stopButton.down ? 90 : 140; easing.type: Easing.OutCubic } }
            Layout.fillWidth: true; Layout.fillHeight: true; text: "Завершить"
            onClicked: if (bar.backend && typeof bar.backend.finishSession === "function") bar.backend.finishSession()
            background: Rectangle { radius: 15; color: stopButton.down ? "#263047" : stopButton.hovered ? "#202c45" : "#131a2e"; border.color: stopButton.activeFocus ? "#f0f2ff" : "#384665"; border.width: stopButton.activeFocus ? 2 : 1; clip: true }
            contentItem: RowLayout {
                anchors.centerIn: parent; spacing: 10
                Canvas { Layout.preferredWidth: 22; Layout.preferredHeight: 22
                    onPaint: {
                        const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                        ctx.strokeStyle = Theme.muted; ctx.lineWidth = 1.8; ctx.strokeRect(3, 3, 16, 16)
                    }
                }
                Text { text: stopButton.text; color: Theme.muted; font.pixelSize: 16; Layout.alignment: Qt.AlignVCenter }
            }
        }
    }
}
