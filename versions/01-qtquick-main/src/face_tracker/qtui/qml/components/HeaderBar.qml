pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: bar
    property string activeTab: "Главное"
    signal tabSelected(string name)

    RowLayout {
        anchors.fill: parent
        spacing: 20
        RowLayout {
            Layout.fillWidth: true
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
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
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
                        Canvas {
                            id: navIcon
                            Layout.preferredWidth: 18; Layout.preferredHeight: 18
                            onPaint: {
                                const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                                ctx.strokeStyle = bar.activeTab === navItem.tabName ? "#ffffff" : "#c3cbe0"
                                ctx.lineWidth = 1.7; ctx.lineJoin = "round"; ctx.lineCap = "round"
                                if (navItem.tabName === "Главное") {
                                    ctx.beginPath(); ctx.moveTo(2, 8); ctx.lineTo(9, 2); ctx.lineTo(16, 8); ctx.moveTo(4, 7); ctx.lineTo(4, 16); ctx.lineTo(8, 16); ctx.lineTo(8, 11); ctx.lineTo(11, 11); ctx.lineTo(11, 16); ctx.lineTo(15, 16); ctx.lineTo(15, 7); ctx.stroke()
                                } else if (navItem.tabName === "История") {
                                    ctx.beginPath(); ctx.arc(9, 9, 7, 0, Math.PI * 2); ctx.stroke()
                                    ctx.beginPath(); ctx.moveTo(9, 4); ctx.lineTo(9, 9); ctx.lineTo(13, 11); ctx.stroke()
                                } else {
                                    ctx.beginPath(); ctx.moveTo(2, 5); ctx.lineTo(9, 2); ctx.lineTo(16, 5); ctx.lineTo(9, 8); ctx.closePath(); ctx.stroke()
                                    ctx.beginPath(); ctx.moveTo(2, 9); ctx.lineTo(9, 12); ctx.lineTo(16, 9); ctx.moveTo(2, 13); ctx.lineTo(9, 16); ctx.lineTo(16, 13); ctx.stroke()
                                }
                            }
                            Connections { target: bar; function onActiveTabChanged() { navIcon.requestPaint() } }
                        }
                        Text { text: navItem.text; color: bar.activeTab === navItem.tabName ? "white" : Theme.text; font.pixelSize: 15; font.weight: Font.Medium; verticalAlignment: Text.AlignVCenter }
                    }
                }
            }
        }

        Rectangle {
            Layout.alignment: Qt.AlignRight
            Layout.preferredWidth: 216
            Layout.preferredHeight: 40
            radius: 20
            color: "#10232b"
            border.color: "#2c817a"
            Row { anchors.centerIn: parent; spacing: 9
                Rectangle { width: 8; height: 8; radius: 4; color: Theme.cyan; anchors.verticalCenter: parent.verticalCenter }
                Canvas {
                    width: 18; height: 20; anchors.verticalCenter: parent.verticalCenter
                    onPaint: {
                        const ctx = getContext("2d"); ctx.clearRect(0, 0, width, height)
                        ctx.strokeStyle = "#e1e9ff"; ctx.lineWidth = 1.6
                        ctx.beginPath(); ctx.arc(9, 7, 4, Math.PI, 0); ctx.lineTo(13, 9); ctx.stroke()
                        ctx.strokeRect(4, 8, 10, 9)
                        ctx.beginPath(); ctx.arc(9, 12, 1, 0, Math.PI * 2); ctx.stroke()
                    }
                }
                Text { text: "Обработка на устройстве"; color: "#cbf6ef"; font.pixelSize: 12 }
            }
        }
    }
}
