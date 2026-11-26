pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: bar
    property string activeTab: "Главное"
    signal tabSelected(string name)

    Item {
        anchors.fill: parent
        RowLayout {
            anchors.left: parent.left
            anchors.top: parent.top
            width: Math.min(parent.width, 720)
            height: 45
            spacing: 12
            Item {
                Layout.preferredWidth: 34; Layout.preferredHeight: 32
                Repeater {
                    model: ["#8c7aff", "#746cff", "#5f83e8"]
                    delegate: Rectangle {
                        required property int index
                        required property string modelData
                        x: index * 7; y: 5 + index * 2; width: 11; height: 22 - index * 2
                        radius: 4; color: modelData; rotation: 22; clip: true
                    }
                }
            }
            Text { text: "КОНТУР"; color: Theme.text; font.pixelSize: 19; font.weight: Font.DemiBold; font.letterSpacing: 1 }
            Rectangle { Layout.preferredWidth: 1; Layout.preferredHeight: 32; color: "#455070"; Layout.leftMargin: 8; Layout.rightMargin: 8 }
            Text { text: "Мимика и самонаблюдение"; color: Theme.text; font.pixelSize: 25; font.weight: Font.Medium; elide: Text.ElideRight; Layout.fillWidth: true }
        }

        Row {
            objectName: "navTabs"
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
            spacing: 8
            Repeater {
                model: ["Главное", "История", "Подробно"]
                delegate: Button {
                    id: navItem
                    required property int index
                    required property string modelData
                    property string tabName: modelData
                    width: navItem.tabName === "Главное" ? 156 : 142
                    height: 42
                    text: navItem.tabName === "Главное" ? "Главная" : navItem.tabName
                    onClicked: bar.tabSelected(navItem.tabName)
                    background: Rectangle {
                        radius: 21; clip: true
                        color: bar.activeTab === navItem.tabName ? "#524cc0" : navItem.hovered ? "#1a2340" : "#10162a"
                        border.color: navItem.activeFocus ? "#f2f4ff" : bar.activeTab === navItem.tabName ? "#8178ff" : "#222d49"
                        border.width: navItem.activeFocus ? 2 : 1
                    }
                    contentItem: RowLayout {
                        spacing: 8
                        Image {
                            id: navIcon
                            Layout.preferredWidth: 18; Layout.preferredHeight: 18
                            source: navItem.tabName === "Главное" ? "../assets/icons/home.svg" : navItem.tabName === "История" ? "../assets/icons/clock.svg" : "../assets/icons/layers.svg"
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                        }
                        Text { text: navItem.text; color: bar.activeTab === navItem.tabName ? "white" : Theme.text; font.pixelSize: 15; font.weight: Font.Medium; verticalAlignment: Text.AlignVCenter }
                    }
                }
            }
        }

        Rectangle {
            objectName: "privacyBadge"
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            width: 216
            height: 40
            radius: 20
            color: "#10232b"
            border.color: "#2c817a"
            Row { anchors.centerIn: parent; spacing: 9
                Rectangle { width: 8; height: 8; radius: 4; color: Theme.cyan; anchors.verticalCenter: parent.verticalCenter }
                Image { width: 18; height: 20; anchors.verticalCenter: parent.verticalCenter; source: "../assets/icons/lock.svg"; fillMode: Image.PreserveAspectFit; smooth: true }
                Text { text: "Обработка на устройстве"; color: "#cbf6ef"; font.pixelSize: 12 }
            }
        }
    }
}
