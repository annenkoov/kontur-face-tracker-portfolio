pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

GlassCard {
    id: card
    property string cameraState: "stopped"
    property string cameraHint: "Подключите камеру, чтобы начать"
    property int frameRevision: 0
    property var landmarkPoints: []
    property var cameraDevices: []
    property int selectedCameraIndex: 0
    property bool cameraScanRunning: false
    signal cameraSelected(int index)
    signal refreshCameras()

    Rectangle {
        anchors.fill: parent
        radius: card.radius
        color: "#080d1b"
        clip: true

        Image {
            id: cameraFrame
            anchors.fill: parent
            anchors.margins: 1
            // Сохраняем исходную геометрию кадра: точки MediaPipe тогда
            // точно совпадают с лицом, а не смещаются из-за обрезки по краям.
            fillMode: Image.PreserveAspectFit
            cache: false
            asynchronous: true
            source: (card.frameRevision > 0 && (card.cameraState === "running" || card.cameraState === "active"))
                    ? "image://camera/latest?rev=" + card.frameRevision : ""
            visible: cameraFrame.status === Image.Ready
        }

        Column {
            anchors.centerIn: parent
            spacing: 14
            visible: cameraFrame.status !== Image.Ready
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: (card.cameraState === "running" || card.cameraState === "active") ? "ОЖИДАНИЕ КАДРА" : card.cameraState === "starting" ? "КАМЕРА ЗАПУСКАЕТСЯ" : "КАМЕРА ОСТАНОВЛЕНА"; color: Theme.text; font.pixelSize: 20; font.weight: Font.Medium }
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: card.cameraHint; color: Theme.muted; font.pixelSize: 14 }
        }

        Canvas {
            id: landmarkOverlay
            anchors.fill: cameraFrame
            visible: cameraFrame.status === Image.Ready && card.landmarkPoints && card.landmarkPoints.length > 0
            onPaint: {
                const ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                ctx.fillStyle = "#a99cff"
                const pts = card.landmarkPoints
                const sourceWidth = cameraFrame.sourceSize.width || width
                const sourceHeight = cameraFrame.sourceSize.height || height
                const scale = Math.min(width / sourceWidth, height / sourceHeight)
                const drawnWidth = sourceWidth * scale, drawnHeight = sourceHeight * scale
                const offsetX = (width - drawnWidth) / 2, offsetY = (height - drawnHeight) / 2
                // Показываем регулярную выборку реальных ключевых точек,
                // не соединяя соседние индексы произвольными линиями.
                for (let i = 0; i < pts.length; i += 4) {
                    const p = pts[i]
                    const x = offsetX + Math.max(0, Math.min(1, Number(p.x))) * drawnWidth
                    const y = offsetY + Math.max(0, Math.min(1, Number(p.y))) * drawnHeight
                    ctx.beginPath(); ctx.arc(x, y, 1.8, 0, Math.PI * 2); ctx.fill()
                }
            }
            onVisibleChanged: requestPaint()
            Connections { target: card; function onLandmarkPointsChanged() { landmarkOverlay.requestPaint() } }
        }

        Button {
            id: cameraPickerButton
            anchors.top: parent.top
            anchors.right: parent.right
            anchors.topMargin: 16
            anchors.rightMargin: 16
            z: 4
            text: card.cameraScanRunning ? "Поиск камер…" : "Выбрать камеру"
            enabled: !card.cameraScanRunning && card.cameraState !== "starting" && card.cameraState !== "running" && card.cameraState !== "active"
            onClicked: cameraPicker.open()
            background: Rectangle {
                radius: 12; color: cameraPickerButton.hovered ? "#333a5b" : "#db11182f"
                border.color: "#7584bd"; clip: true
            }
            contentItem: Text {
                text: cameraPickerButton.text; color: Theme.text; font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                leftPadding: 12; rightPadding: 12
            }
        }

        Popup {
            id: cameraPicker
            parent: Overlay.overlay
            width: 340
            height: Math.min(380, parent ? parent.height - 40 : 380)
            x: parent ? (parent.width - width) / 2 : 0
            y: parent ? (parent.height - height) / 2 : 0
            modal: true
            focus: true
            closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
            background: Rectangle {
                radius: 18; color: "#f0141b33"; border.color: "#7888c9"; clip: true
            }
            contentItem: ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 12
                Text { text: "Выбор камеры"; color: Theme.text; font.pixelSize: 19; font.weight: Font.DemiBold }
                Text {
                    Layout.fillWidth: true
                    text: card.cameraScanRunning ? "Ищем подключённые устройства…" :
                          card.cameraDevices.length ? "Выбери устройство для следующего запуска." : "Подключённые камеры не найдены. Нажми «Обновить список»."
                    color: Theme.muted; wrapMode: Text.WordWrap; font.pixelSize: 13
                }
                ListView {
                    Layout.fillWidth: true; Layout.fillHeight: true; clip: true; spacing: 8
                    model: card.cameraDevices
                    delegate: Button {
                        id: cameraChoice
                        required property var modelData
                        property int deviceIndex: Number(modelData.index)
                        width: ListView.view.width
                        height: 48
                        text: String(cameraChoice.modelData.label || "Камера") +
                              (cameraChoice.deviceIndex === card.selectedCameraIndex ? "  ·  выбрана" : "")
                        onClicked: { card.cameraSelected(cameraChoice.deviceIndex); cameraPicker.close() }
                        background: Rectangle {
                            radius: 12; color: cameraChoice.deviceIndex === card.selectedCameraIndex ? "#4845a4" : "#1c2542"
                            border.color: cameraChoice.deviceIndex === card.selectedCameraIndex ? "#8a82ff" : "#39486d"
                            clip: true
                        }
                        contentItem: Text { text: cameraChoice.text; color: Theme.text; font.pixelSize: 14; leftPadding: 14; verticalAlignment: Text.AlignVCenter }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    Item { Layout.fillWidth: true }
                    Button {
                        id: refreshCameraButton
                        text: "Обновить список"
                        enabled: !card.cameraScanRunning && card.cameraState !== "starting" && card.cameraState !== "running" && card.cameraState !== "active"
                        onClicked: card.refreshCameras()
                        background: Rectangle {
                            radius: 10; color: refreshCameraButton.hovered ? "#343d62" : "#202a49"
                            border.color: "#59699a"; clip: true
                        }
                        contentItem: Text {
                            text: refreshCameraButton.text; color: Theme.text; font.pixelSize: 12
                            horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                            leftPadding: 12; rightPadding: 12
                        }
                    }
                    Button {
                        id: closeCameraButton
                        text: "Закрыть"
                        onClicked: cameraPicker.close()
                        background: Rectangle {
                            radius: 10; color: closeCameraButton.hovered ? "#293451" : "#172039"
                            border.color: "#465579"; clip: true
                        }
                        contentItem: Text {
                            text: closeCameraButton.text; color: Theme.text; font.pixelSize: 12
                            horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                            leftPadding: 12; rightPadding: 12
                        }
                    }
                }
            }
        }

        Rectangle {
            anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom
            anchors.margins: 14; height: 48; radius: 14; color: "#d911172b"; border.color: "#50618b"; clip: true
            RowLayout {
                anchors.fill: parent; anchors.leftMargin: 16; anchors.rightMargin: 16; spacing: 10
                Rectangle { Layout.preferredWidth: 9; Layout.preferredHeight: 9; radius: 5; color: (card.cameraState === "running" || card.cameraState === "active") ? Theme.cyan : "#7d88a7" }
                Text { text: (card.cameraState === "running" || card.cameraState === "active") ? "Камера активна" : "Камера выключена"; color: Theme.text; font.pixelSize: 12; Layout.preferredWidth: 125; elide: Text.ElideRight }
                Rectangle { Layout.preferredWidth: 1; Layout.preferredHeight: 21; color: "#596888" }
                Text { text: "Ключевые точки"; color: Theme.muted; font.pixelSize: 12; Layout.fillWidth: true; elide: Text.ElideRight }
                Text { text: card.landmarkPoints ? String(card.landmarkPoints.length) : "0"; color: Theme.text; font.pixelSize: 12; font.weight: Font.DemiBold }
                Rectangle { Layout.preferredWidth: 1; Layout.preferredHeight: 21; color: "#596888" }
                Text { text: (card.cameraState === "running" || card.cameraState === "active") ? "Локально" : "Ожидаем кадр"; color: Theme.muted; font.pixelSize: 12; Layout.preferredWidth: 92; horizontalAlignment: Text.AlignRight; elide: Text.ElideRight }
            }
        }
    }
}
