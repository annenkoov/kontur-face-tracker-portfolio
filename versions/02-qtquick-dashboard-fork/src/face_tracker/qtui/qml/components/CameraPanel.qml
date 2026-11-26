import QtQuick
import QtQuick.Effects
import QtQuick.Controls
import QtQuick.Layouts

GlassCard {
    id: card
    property string cameraState: "stopped"
    property string cameraHint: "Подключите камеру, чтобы начать"
    property int frameRevision: 0
    property var landmarkPoints: []

    Rectangle {
        id: cameraMask
        anchors.fill: parent
        radius: 16
        visible: false
        layer.enabled: true
    }

    Rectangle {
        id: cameraViewport
        anchors.fill: parent
        radius: 16
        color: "#080d1b"
        clip: true
        layer.enabled: true
        layer.effect: MultiEffect {
            maskEnabled: true
            maskSource: cameraMask
            maskThresholdMin: 0.5
            maskSpreadAtMin: 1.0
        }

        Image {
            id: cameraFrame
            anchors.fill: parent
            anchors.margins: 1
            property bool hasDisplayedFrame: false
            // Сохраняем исходную геометрию кадра: точки MediaPipe тогда
            // точно совпадают с лицом, а не смещаются из-за обрезки по краям.
            fillMode: Image.PreserveAspectFit
            cache: false
            asynchronous: true
            // Новый URL приходит на каждый кадр. Удерживаем предыдущую
            // картинку, пока Qt асинхронно загружает следующую, чтобы не
            // показывать заглушку между обновлениями.
            retainWhileLoading: true
            source: (card.cameraState === "running" || card.cameraState === "active")
                    ? "image://camera/latest?rev=" + card.frameRevision : ""
            visible: cameraFrame.status === Image.Ready ||
                     (cameraFrame.status === Image.Loading && hasDisplayedFrame)
            onStatusChanged: {
                if (status === Image.Ready)
                    hasDisplayedFrame = true
                else if (status === Image.Null || status === Image.Error)
                    hasDisplayedFrame = false
            }
        }

        Column {
            anchors.centerIn: parent
            spacing: 14
            visible: !cameraFrame.hasDisplayedFrame && cameraFrame.status !== Image.Ready
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: (card.cameraState === "running" || card.cameraState === "active") ? "ОЖИДАНИЕ КАДРА" : card.cameraState === "starting" ? "КАМЕРА ЗАПУСКАЕТСЯ" : "КАМЕРА ОСТАНОВЛЕНА"; color: Theme.text; font.pixelSize: 20; font.weight: Font.Medium }
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: card.cameraHint; color: Theme.muted; font.pixelSize: 14 }
        }

        Canvas {
            id: landmarkOverlay
            anchors.fill: cameraFrame
            visible: cameraFrame.visible && card.landmarkPoints && card.landmarkPoints.length > 0
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
