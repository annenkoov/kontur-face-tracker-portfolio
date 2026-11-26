import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window

import "components"

ApplicationWindow {
    id: window
    visible: true
    width: Math.min(1584, Screen.desktopAvailableWidth - 24)
    height: Math.min(990, Screen.desktopAvailableHeight - 36)
    minimumWidth: 1080
    minimumHeight: 700
    title: "Контур — трекер мимических изменений"
    color: Theme.background

    property var backend: null
    property string activeTab: "Главное"

    function chooseTab(name) {
        activeTab = name
        if (backend && typeof backend.selectTab === "function")
            backend.selectTab(name === "Главное" ? "Главная" : name)
        if (name === "История" && backend && typeof backend.refreshHistory === "function")
            backend.refreshHistory()
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#0a0d1a" }
            GradientStop { position: 0.48; color: "#11152b" }
            GradientStop { position: 1.0; color: "#090d19" }
        }
    }

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0; color: "#0b0d19" }
            GradientStop { position: 0.58; color: "#00101424" }
            GradientStop { position: 1; color: "#12192d" }
        }
        opacity: 0.34
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 18

        HeaderBar {
            Layout.fillWidth: true
            Layout.preferredHeight: 76
            activeTab: window.activeTab
            onTabSelected: function(name) { window.chooseTab(name) }
        }

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Flickable {
                id: dashboardScroll
                anchors.fill: parent
                visible: window.activeTab === "Главное"
                clip: true
                contentWidth: width
                contentHeight: Math.max(height, dashboardScroll.width < 1220 ? 1290 : 740)
                ScrollBar.vertical: ScrollBar {
                    policy: dashboardScroll.contentHeight > dashboardScroll.height ? ScrollBar.AsNeeded : ScrollBar.AlwaysOff
                    implicitWidth: 8
                    contentItem: Rectangle { radius: 4; color: "#64739a"; implicitWidth: 6 }
                    background: Rectangle { radius: 4; color: "#19213a"; implicitWidth: 6 }
                }

            GridLayout {
                id: dashboardGrid
                columns: dashboardScroll.width < 1220 ? 1 : 2
                width: dashboardScroll.width - (dashboardScroll.contentHeight > dashboardScroll.height ? 14 : 0)
                height: dashboardScroll.contentHeight
                rowSpacing: 20
                columnSpacing: 20

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: dashboardGrid.columns === 2
                    Layout.preferredWidth: dashboardGrid.columns === 2 ? dashboardGrid.width * 0.528 : -1
                    Layout.preferredHeight: dashboardGrid.columns === 1 ? 510 : -1
                    Layout.minimumWidth: 0
                    spacing: 14

                    CameraPanel {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        cameraState: window.backend ? String(window.backend.cameraState || "stopped") : "stopped"
                        cameraHint: window.backend ? String(window.backend.cameraHint || "Камера пока не подключена") : "Подключите камеру, чтобы начать"
                        frameRevision: window.backend ? Number(window.backend.frameRevision || 0) : 0
                        landmarkPoints: window.backend ? window.backend.landmarkPoints : []
                    }

                    ActionBar {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 92
                        backend: window.backend
                        opacity: window.activeTab === "Главное" ? 1 : 0
                        transform: Translate {
                            y: window.activeTab === "Главное" ? 0 : 8
                            Behavior on y { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
                        }
                        Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: dashboardGrid.columns === 2
                    Layout.preferredWidth: dashboardGrid.columns === 2 ? dashboardGrid.width * 0.472 - dashboardGrid.columnSpacing : -1
                    Layout.preferredHeight: dashboardGrid.columns === 1 ? 760 : -1
                    Layout.minimumWidth: 0
                    spacing: 14
                    opacity: window.activeTab === "Главное" ? 1 : 0
                    transform: Translate {
                        y: window.activeTab === "Главное" ? 0 : 8
                        Behavior on y { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
                    }
                    Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }

                    CurrentStatus {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 124
                        backend: window.backend
                    }

                    StatusGauge {
                        Layout.fillWidth: true
                        Layout.preferredHeight: window.height < 850 ? 205 : 238
                        backend: window.backend
                    }

                    TrendPanel {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        scoreHistory: window.backend ? window.backend.scoreHistory : []
                    }

                    ObservationPanel {
                        Layout.fillWidth: true
                        Layout.preferredHeight: window.height < 850 ? 80 : 96
                        observationText: window.backend ? String(window.backend.observationText || "Создайте личную базу, чтобы начать.") : "Наблюдение появится после запуска"
                    }
                }
            }

            }

            GlassCard {
                id: historyPage
                anchors.fill: parent
                visible: window.activeTab === "История" || opacity > 0.01
                opacity: window.activeTab === "История" ? 1 : 0
                transform: Translate { y: window.activeTab === "История" ? 0 : 8; Behavior on y { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } } }
                Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 24
                    spacing: 14
                    Text { text: "История сессий"; color: Theme.text; font.pixelSize: 24; font.weight: Font.DemiBold }
                    Text { text: "Локальные записи этого устройства"; color: Theme.muted; font.pixelSize: 14 }
                    ListView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: window.backend && window.backend.historyEntries ? window.backend.historyEntries : []
                        delegate: Rectangle {
                            id: entry
                            required property int index
                            required property var modelData
                            width: ListView.view.width
                            height: 58
                            radius: 12
                            color: index % 2 ? "#161d34" : "#121a30"
                            border.color: "#273453"
                            Text {
                                anchors.fill: parent; anchors.leftMargin: 16; anchors.rightMargin: 16
                                verticalAlignment: Text.AlignVCenter
                                text: typeof entry.modelData === "string" ? entry.modelData :
                                      String(entry.modelData.startedAt || entry.modelData.timestamp || "Сессия") +
                                      "   ·   " + String(entry.modelData.state || entry.modelData.summary || "Запись журнала")
                                color: Theme.text; elide: Text.ElideRight; font.pixelSize: 14
                            }
                        }
                        ScrollBar.vertical: ScrollBar { }
                        Text {
                            anchors.centerIn: parent
                            visible: !window.backend || !window.backend.historyEntries || window.backend.historyEntries.length === 0
                            text: "Пока нет сохранённых сессий"
                            color: Theme.muted
                            font.pixelSize: 16
                        }
                    }
                }
            }

            GlassCard {
                id: detailPage
                anchors.fill: parent
                visible: window.activeTab === "Подробно" || opacity > 0.01
                enabled: window.activeTab === "Подробно"
                opacity: window.activeTab === "Подробно" ? 1 : 0
                transform: Translate { y: window.activeTab === "Подробно" ? 0 : 8; Behavior on y { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } } }
                Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
                ScrollView {
                    id: detailsScroll
                    anchors.fill: parent
                    anchors.margins: 24
                    clip: true
                    rightPadding: 14
                    contentWidth: availableWidth
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                        implicitWidth: 8
                        contentItem: Rectangle { radius: 4; color: "#64739a"; implicitWidth: 6 }
                        background: Rectangle { radius: 4; color: "#19213a"; implicitWidth: 6 }
                    }

                    ColumnLayout {
                        width: detailsScroll.availableWidth
                        spacing: 16

                        Text { text: "Подробный анализ"; color: Theme.text; font.pixelSize: 24; font.weight: Font.DemiBold }
                        Text {
                            Layout.fillWidth: true
                            text: "Изменения лица сравниваются с вашей личной базой."
                            color: Theme.muted
                            wrapMode: Text.WordWrap
                            font.pixelSize: 14
                        }
                        FaceSignalsPanel {
                            Layout.fillWidth: true
                            backend: window.backend
                        }
                        Button {
                            text: "Приостановить сессию"
                            onClicked: if (window.backend && typeof window.backend.pauseSession === "function") window.backend.pauseSession()
                            background: Rectangle { radius: 12; color: "#1a2340"; border.color: "#596a9c"; clip: true }
                            contentItem: Text { text: "Приостановить сессию"; color: Theme.text; font.pixelSize: 13; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                        }
                        PersonalLabelsPanel {
                            Layout.fillWidth: true
                            backend: window.backend
                        }
                    }
                }
            }
        }
    }
}
