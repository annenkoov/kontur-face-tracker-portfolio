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

            RowLayout {
                anchors.fill: parent
                spacing: 20
                visible: window.activeTab === "Главное"

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: 1.05
                    spacing: 14

                    CameraPanel {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        cameraState: window.backend ? String(window.backend.cameraState || "stopped") : "stopped"
                        cameraHint: window.backend ? String(window.backend.cameraHint || "Камера пока не подключена") : "Подключите камеру, чтобы начать"
                        frameRevision: window.backend ? Number(window.backend.frameRevision || 0) : 0
                        landmarkPoints: window.backend ? window.backend.landmarkPoints : []
                        cameraDevices: window.backend ? window.backend.cameraDevices : []
                        selectedCameraIndex: window.backend ? Number(window.backend.selectedCameraIndex || 0) : 0
                        cameraScanRunning: window.backend ? Boolean(window.backend.cameraScanRunning) : false
                        onCameraSelected: function(index) { if (window.backend) window.backend.selectCamera(index) }
                        onRefreshCameras: if (window.backend) window.backend.refreshCameras()
                    }

                    ActionBar {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 92
                        backend: window.backend
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: 1
                    spacing: 14

                    CurrentStatus {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 124
                        backend: window.backend
                    }

                    StatusGauge {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 238
                        backend: window.backend
                    }

                    TrendPanel {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        scoreHistory: window.backend ? window.backend.scoreHistory : []
                    }

                    ObservationPanel {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 96
                        observationText: window.backend ? String(window.backend.observationText || "Создайте личную базу, чтобы начать.") : "Наблюдение появится после запуска"
                    }
                }
            }

            GlassCard {
                anchors.fill: parent
                visible: window.activeTab === "История"
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
                anchors.fill: parent
                visible: window.activeTab === "Подробно"
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 24
                    spacing: 12
                    Text { text: "Подробный анализ"; color: Theme.text; font.pixelSize: 24; font.weight: Font.DemiBold }
                    Text { text: "Отслеживание основано на изменениях ключевых точек лица относительно вашей личной базы."; color: Theme.muted; wrapMode: Text.WordWrap; font.pixelSize: 14 }
                    Text { text: "Распознавание мимики: готовые модели HSEmotion + FER+ (8 выражений, включая грусть). Веса хранятся локально; свои примеры собирать не нужно."; color: Theme.text; wrapMode: Text.WordWrap; font.pixelSize: 14 }
                    Text { text: "Личная калибровка нужна только для исходного уровня мимики. Вероятности выражений не являются диагнозом и отдельно не меняют индекс."; color: Theme.muted; wrapMode: Text.WordWrap; font.pixelSize: 13 }
                    Button {
                        text: "Приостановить сессию"
                        onClicked: if (window.backend && typeof window.backend.pauseSession === "function") window.backend.pauseSession()
                        background: Rectangle { radius: 12; color: "#1a2340"; border.color: "#596a9c"; clip: true }
                        contentItem: Text { text: "Приостановить сессию"; color: Theme.text; font.pixelSize: 13; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                    }
                    Item { Layout.fillHeight: true }
                    Text { Layout.alignment: Qt.AlignHCenter; text: "Две предобученные модели · локальный анализ · без ручной разметки"; color: Theme.muted; font.pixelSize: 15 }
                    Item { Layout.fillHeight: true }
                }
            }
        }
    }
}
