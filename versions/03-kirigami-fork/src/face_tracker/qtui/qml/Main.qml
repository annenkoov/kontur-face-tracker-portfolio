import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import org.kde.kirigami as Kirigami

import "components"

Kirigami.AbstractApplicationWindow {
    id: window
    controlsVisible: false
    visible: true
    width: Math.min(1584, Screen.desktopAvailableWidth - 24)
    height: Math.min(990, Screen.desktopAvailableHeight - 36)
    minimumWidth: 1080
    minimumHeight: 700
    title: "Контур — трекер мимических изменений"
    color: Theme.background

    property var backend: null
    property bool reducedMotion: true
    property string activeTab: "Главное"
    readonly property real uiScale: Math.min(1, width / 1350, height / 850)

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
        id: dashboardLayout
        x: 24
        y: 24
        width: (window.width - 48) / window.uiScale
        height: (window.height - 48) / window.uiScale
        scale: window.uiScale
        transformOrigin: Item.TopLeft
        spacing: 18

        HeaderBar {
                        objectName: "header"
            Layout.fillWidth: true
            Layout.preferredHeight: 108
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
                    spacing: dashboardLayout.height < 800 ? 10 : 14

                    CameraPanel {
                        objectName: "cameraPanel"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        cameraState: window.backend ? String(window.backend.cameraState || "stopped") : "stopped"
                        cameraHint: window.backend ? String(window.backend.cameraHint || "Камера пока не подключена") : "Подключите камеру, чтобы начать"
                        frameRevision: window.backend ? Number(window.backend.frameRevision || 0) : 0
                        landmarkPoints: window.backend ? window.backend.landmarkPoints : []
                        backend: window.backend
                    }

                    ActionBar {
                        objectName: "actionBar"
                        reducedMotion: window.reducedMotion
                        Layout.fillWidth: true
                        Layout.preferredHeight: dashboardLayout.height < 800 ? 76 : 92
                        backend: window.backend
                    }
                }

                ColumnLayout {
                    opacity: window.activeTab === "Главное" ? 1 : 0
                    Behavior on opacity { enabled: !window.reducedMotion; NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: 1
                    spacing: dashboardLayout.height < 800 ? 10 : 14

                    CurrentStatus {
                        objectName: "currentStatus"
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.max(100, Math.min(124, dashboardLayout.height * 0.14))
                        backend: window.backend
                    }

                    StatusGauge {
                        objectName: "statusGauge"
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.max(195, Math.min(238, dashboardLayout.height * 0.27))
                        backend: window.backend
                    }

                    TrendPanel {
                        objectName: "trendPanel"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        scoreHistory: window.backend ? window.backend.scoreHistory : []
                        hasScore: window.backend ? window.backend.hasScore : false
                        scoreValue: window.backend ? window.backend.scoreValue : 0
                        collectingBaseline: window.backend ? window.backend.stateTitle === "Калибровка личной базы" : false
                    }

                    ObservationPanel {
                        objectName: "observationPanel"
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.max(85, Math.min(96, dashboardLayout.height * 0.11))
                        observationText: window.backend ? String(window.backend.observationText || "Создайте личную базу, чтобы начать.") : "Наблюдение появится после запуска"
                    }
                }
            }

            GlassCard {
                id: historyPage
                objectName: "historyPage"
                anchors.fill: parent
                visible: window.activeTab === "История"
                transform: Translate { y: historyPage.visible ? 0 : 8; Behavior on y { enabled: !window.reducedMotion; NumberAnimation { duration: 160; easing.type: Easing.OutCubic } } }
                opacity: visible ? 1 : 0
                Behavior on opacity { enabled: !window.reducedMotion; NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
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
                id: detailsPage
                objectName: "detailsPage"
                anchors.fill: parent
                visible: window.activeTab === "Подробно"
                transform: Translate { y: detailsPage.visible ? 0 : 8; Behavior on y { enabled: !window.reducedMotion; NumberAnimation { duration: 160; easing.type: Easing.OutCubic } } }
                opacity: visible ? 1 : 0
                Behavior on opacity { enabled: !window.reducedMotion; NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
                ScrollView {
                    anchors.fill: parent
                    anchors.margins: 24
                    clip: true
                    contentWidth: availableWidth
                    ColumnLayout {
                        width: parent.width
                        spacing: 14
                        Text { text: "Подробный анализ"; color: Theme.text; font.pixelSize: 24; font.weight: Font.DemiBold }
                        Text {
                            Layout.fillWidth: true
                            text: "Индекс показывает устойчивое изменение мимики относительно твоей личной базы. " +
                                  "Это не медицинская оценка стресса."
                            color: Theme.muted; wrapMode: Text.WordWrap; font.pixelSize: 14
                        }
                        GlassCard {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 300
                            contentMargin: 18
                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 10
                                Text { text: "ЛИЧНАЯ ПРОВЕРКА ЭМОЦИЙ"; color: Theme.violet; font.pixelSize: 13; font.weight: Font.Bold }
                                Text {
                                    Layout.fillWidth: true
                                    text: "Модель: " + (window.backend ? String(window.backend.emotionModelStatus) : "HSEmotion")
                                    color: Theme.text; font.pixelSize: 15; font.weight: Font.DemiBold; wrapMode: Text.WordWrap
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: window.backend ? String(window.backend.emotionSampleCounts) : "Примеры не сохранены"
                                    color: Theme.muted; font.pixelSize: 12; wrapMode: Text.WordWrap
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: window.backend ? String(window.backend.emotionStatus) : "Локальная проверка"
                                    color: Theme.muted; font.pixelSize: 12; wrapMode: Text.WordWrap
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 10
                                    Button {
                                        id: compareEmotionButton
                                        Layout.fillWidth: true
                                        text: window.backend && window.backend.emotionTaskRunning ? "Считаю…" : "Сравнить примеры"
                                        enabled: window.backend && !window.backend.emotionTaskRunning
                                        onClicked: window.backend.compareEmotionModels()
                                        background: Rectangle { radius: 11; color: compareEmotionButton.down ? "#5750a5" : compareEmotionButton.hovered ? "#343b62" : "#252d4a"; border.color: "#6879ab"; clip: true }
                                        contentItem: Text { text: compareEmotionButton.text; color: Theme.text; font.pixelSize: 13; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                                    }
                                    Button {
                                        id: adaptEmotionButton
                                        Layout.fillWidth: true
                                        text: window.backend && window.backend.emotionTaskRunning ? "Проверяю…" : "Оценить адаптацию"
                                        enabled: window.backend && !window.backend.emotionTaskRunning
                                        onClicked: window.backend.evaluateEmotionAdaptation()
                                        background: Rectangle { radius: 11; color: adaptEmotionButton.down ? "#7770ed" : adaptEmotionButton.hovered ? "#9189ff" : "#7469ed"; border.color: "#aaa2ff"; clip: true }
                                        contentItem: Text { text: adaptEmotionButton.text; color: "white"; font.pixelSize: 13; font.weight: Font.DemiBold; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                                    }
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: "Показывай метки сам. Для личной поправки нужны повторные примеры в разных сессиях; она включится лишь если пройдёт отдельную проверку."
                                    color: Theme.muted; font.pixelSize: 11; wrapMode: Text.WordWrap
                                }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Button {
                                id: resetAdapterButton
                                text: "Сбросить личную поправку"
                                enabled: window.backend && String(window.backend.emotionModelStatus).startsWith("Персональная") && !window.backend.emotionTaskRunning
                                onClicked: window.backend.resetEmotionAdaptation()
                                background: Rectangle { radius: 10; color: "#1a2340"; border.color: "#596a9c"; clip: true }
                                contentItem: Text { text: resetAdapterButton.text; color: Theme.text; font.pixelSize: 12; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                            }
                            Button {
                                id: deleteSamplesButton
                                text: "Удалить примеры"
                                enabled: window.backend && !window.backend.emotionTaskRunning
                                onClicked: deleteEmotionDialog.open()
                                background: Rectangle { radius: 10; color: "#30202f"; border.color: "#8d526e"; clip: true }
                                contentItem: Text { text: deleteSamplesButton.text; color: "#ffc0d2"; font.pixelSize: 12; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                            }
                            Item { Layout.fillWidth: true }
                            Button {
                                id: pauseSessionButton
                                text: "Приостановить сессию"
                                onClicked: if (window.backend) window.backend.pauseSession()
                                background: Rectangle { radius: 10; color: "#1a2340"; border.color: "#596a9c"; clip: true }
                                contentItem: Text { text: pauseSessionButton.text; color: Theme.text; font.pixelSize: 12; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                            }
                        }
                    }
                }

                Dialog {
                    id: deleteEmotionDialog
                    parent: Overlay.overlay
                    anchors.centerIn: parent
                    width: Math.min(430, parent.width - 48)
                    modal: true
                    padding: 20
                    background: Rectangle { radius: 18; color: "#f21a2139"; border.color: "#7787bd"; border.width: 1 }
                    contentItem: ColumnLayout {
                        spacing: 12
                        Text { text: "Удалить локальные примеры?"; color: Theme.text; font.pixelSize: 19; font.weight: Font.DemiBold }
                        Text { text: "Удалятся вручную сохранённые обрезки лица, отчёты и персональная поправка. Общая модель останется."; color: Theme.muted; font.pixelSize: 13; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        RowLayout {
                            Layout.fillWidth: true
                            Item { Layout.fillWidth: true }
                            Button { text: "Отмена"; onClicked: deleteEmotionDialog.close() }
                            Button {
                                text: "Удалить"
                                onClicked: { if (window.backend) window.backend.deleteEmotionSamples(); deleteEmotionDialog.close() }
                                background: Rectangle { radius: 10; color: "#7b3857"; border.color: "#bd668b" }
                                contentItem: Text { text: "Удалить"; color: "white"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                            }
                        }
                    }
                }
            }
        }
    }
}
