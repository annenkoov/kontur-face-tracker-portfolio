import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

GlassCard {
    id: card
    lightColor: "#8b6fff"
    lightStrength: 0.22
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
                Image { Layout.preferredWidth: 44; Layout.preferredHeight: 44; source: "../assets/icons/face.svg"; fillMode: Image.PreserveAspectFit; smooth: true }
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
                Image { anchors.fill: parent; source: "../assets/icons/target.svg"; fillMode: Image.PreserveAspectFit; smooth: true }
            }
        }
    }
}
