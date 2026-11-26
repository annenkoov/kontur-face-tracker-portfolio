import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

GlassCard {
    id: bar
    property var backend: null
    property bool reducedMotion: true

    RowLayout {
        anchors.fill: parent
        spacing: 12
        Button {
            id: createButton
            objectName: "createButton"
            Layout.fillWidth: true; Layout.fillHeight: true; text: "Создать базу"
            onClicked: if (bar.backend && typeof bar.backend.createBaseline === "function") bar.backend.createBaseline()
            background: Rectangle { radius: 15; color: createButton.down ? "#26345b" : createButton.hovered ? "#202c49" : "#151e36"; border.color: createButton.activeFocus ? "#f0f2ff" : "#6275aa"; border.width: createButton.activeFocus ? 2 : 1; clip: true; Behavior on color { enabled: !bar.reducedMotion; ColorAnimation { duration: 120 } } }
            contentItem: RowLayout {
                anchors.centerIn: parent; spacing: 10
                Image { Layout.preferredWidth: 24; Layout.preferredHeight: 24; source: "../assets/icons/layers.svg"; fillMode: Image.PreserveAspectFit; smooth: true }
                Text { text: createButton.text; color: Theme.text; font.pixelSize: 18; Layout.alignment: Qt.AlignVCenter }
            }
        }
        Button {
            id: startButton
            objectName: "startButton"
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
                opacity: startButton.down ? 0.78 : startButton.hovered ? 0.92 : 1
                Behavior on opacity { enabled: !bar.reducedMotion; NumberAnimation { duration: 120 } }
                clip: true
            }
            contentItem: RowLayout {
                anchors.centerIn: parent; spacing: 10
                Image { Layout.preferredWidth: 24; Layout.preferredHeight: 24; source: "../assets/icons/play.svg"; fillMode: Image.PreserveAspectFit; smooth: true }
                Text { text: startButton.text; color: "white"; font.pixelSize: 19; font.weight: Font.DemiBold; Layout.alignment: Qt.AlignVCenter }
            }
        }
        Button {
            id: stopButton
            objectName: "stopButton"
            Layout.fillWidth: true; Layout.fillHeight: true; text: "Завершить"
            onClicked: if (bar.backend && typeof bar.backend.finishSession === "function") bar.backend.finishSession()
            background: Rectangle { radius: 15; color: stopButton.down ? "#263047" : stopButton.hovered ? "#202c45" : "#131a2e"; border.color: stopButton.activeFocus ? "#f0f2ff" : "#384665"; border.width: stopButton.activeFocus ? 2 : 1; clip: true; Behavior on color { enabled: !bar.reducedMotion; ColorAnimation { duration: 120 } } }
            contentItem: RowLayout {
                anchors.centerIn: parent; spacing: 10
                Image { Layout.preferredWidth: 24; Layout.preferredHeight: 24; source: "../assets/icons/stop.svg"; fillMode: Image.PreserveAspectFit; smooth: true }
                Text { text: stopButton.text; color: Theme.muted; font.pixelSize: 18; Layout.alignment: Qt.AlignVCenter }
            }
        }
    }
}
