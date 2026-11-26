import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window

GlassCard {
    id: card
    property string cameraState: "stopped"
    property string cameraHint: "Подключите камеру, чтобы начать"
    property int frameRevision: 0
    property var landmarkPoints: []
    property var backend: null
    // Loading очередного асинхронного кадра не прячет уже показанное изображение.
    property bool hasRenderedFrame: false
    contentMargin: 1

    Rectangle {
        anchors.fill: parent
        radius: card.radius
        color: "#080d1b"
        clip: true

        Image {
            id: cameraFrame
            objectName: "cameraFrame"
            anchors.fill: parent
            anchors.margins: 1
            // Сохраняем исходную геометрию кадра: точки MediaPipe тогда
            // точно совпадают с лицом, а не смещаются из-за обрезки по краям.
            fillMode: Image.PreserveAspectFit
            cache: false
            asynchronous: true
            // Qt 6.12: сохраняем предыдущую текстуру до готовности следующего URL.
            retainWhileLoading: true
            source: (card.cameraState === "running" || card.cameraState === "active")
                    ? "image://camera/latest?rev=" + card.frameRevision : ""
            visible: card.hasRenderedFrame
            onStatusChanged: {
                // На Image.Loading оставляем последнюю удачную текстуру на экране.
                // Пустой Ready приходит только после того, как контроллер очистил
                // кадр из-за подтверждённой устарелости.
                if (status === Image.Ready) {
                    card.hasRenderedFrame = sourceSize.width > 0 && sourceSize.height > 0
                }
            }
        }

        ToolButton {
            id: cameraSettingsButton
            objectName: "cameraSettingsButton"
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 16
            width: 48
            height: 48
            z: 12
            focusPolicy: Qt.StrongFocus
            Accessible.name: "Выбрать камеру"
            ToolTip.visible: hovered
            ToolTip.text: "Выбрать камеру"
            onClicked: cameraDialog.open()
            background: Rectangle {
                radius: 14
                color: cameraSettingsButton.down ? "#554f82db" : cameraSettingsButton.hovered ? "#403c5276" : "#cc151c34"
                border.width: cameraSettingsButton.activeFocus ? 2 : 1
                border.color: cameraSettingsButton.activeFocus ? "#c4bdff" : "#6878a6"
                Behavior on color { ColorAnimation { duration: 120 } }
            }
            contentItem: Canvas {
                anchors.fill: parent
                onPaint: {
                    const ctx = getContext("2d")
                    ctx.clearRect(0, 0, width, height)
                    ctx.strokeStyle = "#f2f3ff"
                    ctx.lineWidth = 2
                    ctx.lineJoin = "round"
                    ctx.lineCap = "round"
                    // Простая контурная камера вместо символа-шрифта: значок
                    // одинаково отображается на разных версиях Windows.
                    ctx.beginPath(); ctx.roundRect(width * .22, height * .31, width * .43, height * .38, 4); ctx.stroke()
                    ctx.beginPath(); ctx.moveTo(width * .65, height * .40); ctx.lineTo(width * .79, height * .31)
                    ctx.lineTo(width * .79, height * .69); ctx.lineTo(width * .65, height * .60); ctx.stroke()
                    ctx.fillStyle = "#a99cff"
                    ctx.beginPath(); ctx.arc(width * .43, height * .50, 2.4, 0, Math.PI * 2); ctx.fill()
                }
            }
        }

        ToolButton {
            id: labelEmotionButton
            objectName: "labelEmotionButton"
            anchors.right: cameraSettingsButton.left
            anchors.rightMargin: 8
            anchors.top: parent.top
            anchors.margins: 16
            width: 48
            height: 48
            z: 12
            focusPolicy: Qt.StrongFocus
            Accessible.name: "Отметить выражение лица"
            ToolTip.visible: hovered
            ToolTip.text: "Сохранить личный пример выражения"
            onClicked: emotionLabelDialog.open()
            background: Rectangle {
                radius: 14
                color: labelEmotionButton.down ? "#554f82db" : labelEmotionButton.hovered ? "#403c5276" : "#cc151c34"
                border.width: labelEmotionButton.activeFocus ? 2 : 1
                border.color: labelEmotionButton.activeFocus ? "#c4bdff" : "#6878a6"
            }
            contentItem: Canvas {
                anchors.fill: parent
                onPaint: {
                    const ctx = getContext("2d")
                    ctx.clearRect(0, 0, width, height)
                    ctx.strokeStyle = "#f2f3ff"
                    ctx.lineWidth = 1.8
                    ctx.lineCap = "round"
                    ctx.beginPath(); ctx.arc(width * .5, height * .5, width * .27, 0, Math.PI * 2); ctx.stroke()
                    ctx.fillStyle = "#a99cff"
                    ctx.beginPath(); ctx.arc(width * .41, height * .46, 1.8, 0, Math.PI * 2); ctx.fill()
                    ctx.beginPath(); ctx.arc(width * .59, height * .46, 1.8, 0, Math.PI * 2); ctx.fill()
                    ctx.beginPath(); ctx.moveTo(width * .40, height * .60)
                    ctx.quadraticCurveTo(width * .50, height * .67, width * .60, height * .60); ctx.stroke()
                }
            }
        }

        Dialog {
            id: emotionLabelDialog
            objectName: "emotionLabelDialog"
            parent: Overlay.overlay
            anchors.centerIn: parent
            width: Math.min(460, parent.width - 48)
            modal: true
            focus: true
            title: "Личный пример выражения"
            standardButtons: Dialog.NoButton
            padding: 22
            background: Rectangle {
                radius: 20
                color: "#f21a2139"
                border.width: 1
                border.color: "#7787bd"
                gradient: Gradient {
                    GradientStop { position: 0; color: "#1c8276ff" }
                    GradientStop { position: 0.56; color: "#0bffffff" }
                    GradientStop { position: 1; color: "#153de1d4" }
                }
            }
            contentItem: ColumnLayout {
                spacing: 14
                Text {
                    text: "Какое выражение ты показываешь?"
                    color: Theme.text
                    font.pixelSize: 20
                    font.weight: Font.DemiBold
                    Layout.fillWidth: true
                }
                Text {
                    text: "Прими выражение, выбери метку и нажми «Сохранить пример». " +
                          "Сохранится только один обрезанный кадр после твоего нажатия."
                    color: Theme.muted
                    font.pixelSize: 13
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
                ComboBox {
                    id: emotionLabelCombo
                    objectName: "emotionLabelCombo"
                    Layout.fillWidth: true
                    model: card.backend ? card.backend.emotionLabels : []
                    textRole: "title"
                    valueRole: "value"
                    currentIndex: indexOfValue("Neutral")
                    background: Rectangle {
                        radius: 11
                        color: "#cc10172c"
                        border.width: emotionLabelCombo.activeFocus ? 2 : 1
                        border.color: emotionLabelCombo.activeFocus ? "#aaa1ff" : "#596c9c"
                    }
                    contentItem: Text {
                        leftPadding: 14
                        rightPadding: emotionLabelCombo.indicator.width + 12
                        text: emotionLabelCombo.displayText
                        color: Theme.text
                        verticalAlignment: Text.AlignVCenter
                        font.pixelSize: 14
                    }
                    indicator: Canvas {
                        width: 16
                        height: 12
                        x: emotionLabelCombo.width - width - 14
                        y: (emotionLabelCombo.height - height) / 2
                        onPaint: {
                            const ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)
                            ctx.strokeStyle = "#43dfd4"
                            ctx.lineWidth = 2
                            ctx.lineCap = "round"
                            ctx.beginPath(); ctx.moveTo(3, 3); ctx.lineTo(8, 8); ctx.lineTo(13, 3); ctx.stroke()
                        }
                    }
                }
                Text {
                    Layout.fillWidth: true
                    text: card.backend ? String(card.backend.emotionSampleCounts) : "Примеры хранятся локально."
                    color: Theme.muted
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                }
                Text {
                    Layout.fillWidth: true
                    text: card.backend ? String(card.backend.emotionStatus) : "Примеров пока нет."
                    color: Theme.muted
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Button {
                        id: saveEmotionButton
                        Layout.fillWidth: true
                        text: card.cameraState === "active" ? "Сохранить пример" : "Нужна активная сессия"
                        enabled: card.backend && card.cameraState === "active" && emotionLabelCombo.currentIndex >= 0
                        onClicked: {
                            if (card.backend && emotionLabelCombo.currentIndex >= 0)
                                card.backend.saveEmotionSample(String(emotionLabelCombo.currentValue))
                        }
                        background: Rectangle {
                            radius: 11
                            color: saveEmotionButton.down ? "#7770ed" : saveEmotionButton.hovered ? "#9189ff" : "#7469ed"
                            border.color: "#aaa2ff"
                            border.width: 1
                        }
                        contentItem: Text {
                            text: saveEmotionButton.text
                            color: saveEmotionButton.enabled ? "white" : Theme.muted
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                    Button {
                        text: "Отмена"
                        onClicked: emotionLabelDialog.close()
                        background: Rectangle {
                            radius: 11
                            color: "#302b385f"
                            border.color: "#7778a7ee"
                            border.width: 1
                        }
                        contentItem: Text {
                            text: "Отмена"
                            color: Theme.text
                            font.pixelSize: 13
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }
            Connections {
                target: card.backend
                function onEmotionSampleCountsChanged() { emotionLabelDialog.close() }
            }
        }

        Column {
            anchors.centerIn: parent
            spacing: 14
            visible: !card.hasRenderedFrame
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: (card.cameraState === "running" || card.cameraState === "active") ? "ОЖИДАНИЕ КАДРА" : card.cameraState === "starting" ? "КАМЕРА ЗАПУСКАЕТСЯ" : "КАМЕРА ОСТАНОВЛЕНА"; color: Theme.text; font.pixelSize: 20; font.weight: Font.Medium }
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: card.cameraHint; color: Theme.muted; font.pixelSize: 14 }
        }

        Connections {
            target: card
            function onCameraStateChanged() {
                if (card.cameraState !== "running" && card.cameraState !== "active")
                    card.hasRenderedFrame = false
            }
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

        Dialog {
            id: cameraDialog
            objectName: "cameraDialog"
            parent: Overlay.overlay
            anchors.centerIn: parent
            width: Math.min(430, parent.width - 48)
            modal: true
            focus: true
            title: "Выбор камеры"
            standardButtons: Dialog.NoButton
            padding: 22
            onOpened: {
                if (card.backend && card.backend.cameraScanState === "idle")
                    card.backend.refreshCameras()
            }
            background: Rectangle {
                radius: 20
                color: "#f21a2139"
                border.width: 1
                border.color: "#7787bd"
                Rectangle {
                    anchors.fill: parent
                    anchors.margins: 1
                    radius: 19
                    color: "transparent"
                    gradient: Gradient {
                        GradientStop { position: 0; color: "#1c8276ff" }
                        GradientStop { position: 0.56; color: "#0bffffff" }
                        GradientStop { position: 1; color: "#153de1d4" }
                    }
                }
            }
            contentItem: ColumnLayout {
                spacing: 14
                Text {
                    text: "Источник видео"
                    color: Theme.text
                    font.pixelSize: 20
                    font.weight: Font.DemiBold
                    Layout.fillWidth: true
                }
                Text {
                    text: card.cameraState === "running" || card.cameraState === "active"
                          ? "Останови обработку, чтобы сменить камеру."
                          : "Выбранная камера будет использоваться при следующем запуске."
                    color: Theme.muted
                    font.pixelSize: 13
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
                ComboBox {
                    id: cameraCombo
                    objectName: "cameraCombo"
                    Layout.fillWidth: true
                    model: card.backend ? card.backend.cameraDevices : []
                    textRole: "label"
                    valueRole: "index"
                    enabled: card.backend && card.cameraState !== "running" && card.cameraState !== "active" && card.cameraState !== "starting" && card.backend.cameraScanState !== "scanning" && count > 0
                    currentIndex: card.backend ? indexOfValue(Number(card.backend.selectedCameraIndex)) : -1
                    onActivated: function(comboIndex) {
                        if (card.backend && comboIndex >= 0)
                            card.backend.selectCamera(Number(valueAt(comboIndex)))
                    }
                    background: Rectangle {
                        radius: 11
                        color: "#cc10172c"
                        border.width: cameraCombo.activeFocus ? 2 : 1
                        border.color: cameraCombo.activeFocus ? "#aaa1ff" : "#596c9c"
                    }
                    contentItem: Text {
                        leftPadding: 14
                        rightPadding: cameraCombo.indicator.width + 12
                        text: {
                            if (cameraCombo.displayText)
                                return cameraCombo.displayText
                            const state = card.backend ? String(card.backend.cameraScanState) : "idle"
                            if (state === "scanning") return "Поиск устройств…"
                            if (state === "ready" || state === "empty")
                                return "Камера " + (Number(card.backend.selectedCameraIndex) + 1) + " недоступна"
                            if (state === "error") return "Список не проверен"
                            return "Камера " + (Number(card.backend ? card.backend.selectedCameraIndex : 0) + 1)
                        }
                        color: cameraCombo.enabled ? Theme.text : Theme.muted
                        verticalAlignment: Text.AlignVCenter
                        font.pixelSize: 14
                        elide: Text.ElideRight
                    }
                    indicator: Canvas {
                        width: 16
                        height: 12
                        x: cameraCombo.width - width - 14
                        y: (cameraCombo.height - height) / 2
                        onPaint: {
                            const ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)
                            ctx.strokeStyle = "#43dfd4"
                            ctx.lineWidth = 2
                            ctx.lineCap = "round"
                            ctx.beginPath(); ctx.moveTo(3, 3); ctx.lineTo(8, 8); ctx.lineTo(13, 3); ctx.stroke()
                        }
                    }
                }
                Text {
                    Layout.fillWidth: true
                    color: card.backend && card.backend.cameraScanState === "error" ? "#ff9aab" : Theme.muted
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                    text: {
                        const state = card.backend ? String(card.backend.cameraScanState) : "idle"
                        if (state === "scanning") return "Ищем подключённые камеры…"
                        if (state === "ready") {
                            if (cameraCombo.indexOfValue(Number(card.backend.selectedCameraIndex)) < 0)
                                return "Найдено устройств: " + cameraCombo.count + ". Сохранённая камера не найдена — выбери другую."
                            return "Найдено устройств: " + cameraCombo.count
                        }
                        if (state === "empty") return "Камеры не найдены. Проверь подключение и повтори поиск."
                        if (state === "error") return "Не удалось проверить список. Повтори поиск."
                        return "Нажми «Обновить список», чтобы найти камеры."
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Button {
                        id: refreshButton
                        Layout.fillWidth: true
                        text: card.backend && card.backend.cameraScanState === "scanning" ? "Поиск…" : "Обновить список"
                        enabled: card.backend && card.cameraState !== "running" && card.cameraState !== "active" && card.cameraState !== "starting" && card.backend.cameraScanState !== "scanning"
                        onClicked: card.backend.refreshCameras()
                        background: Rectangle {
                            radius: 11
                            color: refreshButton.down ? "#5854bbff" : refreshButton.hovered ? "#474574cf" : "#302b385f"
                            border.color: "#7778a7ee"
                            border.width: 1
                        }
                        contentItem: Text {
                            text: refreshButton.text
                            color: refreshButton.enabled ? Theme.text : Theme.muted
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                    Button {
                        id: doneButton
                        text: "Готово"
                        onClicked: cameraDialog.close()
                        background: Rectangle {
                            radius: 11
                            color: doneButton.down ? "#7770ed" : doneButton.hovered ? "#9189ff" : "#7469ed"
                            border.color: "#aaa2ff"
                            border.width: 1
                        }
                        contentItem: Text {
                            text: doneButton.text
                            color: "white"
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }
        }

        // Статичные уголки скрывают прямоугольные углы видеослоя без
        // постобработки каждого кадра и без повторной выборки изображения.
        Canvas {
            id: cornerMask
            anchors.fill: parent
            z: 10
            onWidthChanged: requestPaint()
            onHeightChanged: requestPaint()
            onPaint: {
                const ctx = getContext("2d"), r = card.radius, w = width, h = height
                ctx.clearRect(0, 0, w, h)
                ctx.save(); ctx.beginPath()
                ctx.moveTo(0, r); ctx.lineTo(0, 0); ctx.lineTo(r, 0); ctx.quadraticCurveTo(0, 0, 0, r); ctx.closePath()
                ctx.moveTo(w-r, 0); ctx.lineTo(w, 0); ctx.lineTo(w, r); ctx.quadraticCurveTo(w, 0, w-r, 0); ctx.closePath()
                ctx.moveTo(0, h-r); ctx.lineTo(0, h); ctx.lineTo(r, h); ctx.quadraticCurveTo(0, h, 0, h-r); ctx.closePath()
                ctx.moveTo(w-r, h); ctx.lineTo(w, h); ctx.lineTo(w, h-r); ctx.quadraticCurveTo(w, h, w-r, h); ctx.closePath()
                ctx.clip()
                // Повторяем два градиента окна в его координатах, чтобы
                // уголки не превращались в однотонные квадратные пятна.
                const win = card.Window.window
                const origin = mapToItem(null, 0, 0), unit = mapToItem(null, 1, 1)
                const sx = unit.x - origin.x, sy = unit.y - origin.y
                if (win && sx > 0 && sy > 0) {
                    const vertical = ctx.createLinearGradient(0, -origin.y/sy, 0, (win.height-origin.y)/sy)
                    vertical.addColorStop(0, "#0a0d1a"); vertical.addColorStop(0.48, "#11152b"); vertical.addColorStop(1, "#090d19")
                    ctx.fillStyle = vertical; ctx.fillRect(0, 0, w, h)
                    const horizontal = ctx.createLinearGradient(-origin.x/sx, 0, (win.width-origin.x)/sx, 0)
                    horizontal.addColorStop(0, "rgba(11,13,25,0.34)"); horizontal.addColorStop(0.58, "rgba(16,20,36,0)"); horizontal.addColorStop(1, "rgba(18,25,45,0.34)")
                    ctx.fillStyle = horizontal; ctx.fillRect(0, 0, w, h)
                } else { ctx.fillStyle = Theme.background; ctx.fillRect(0, 0, w, h) }
                ctx.restore()
            }
            Connections { target: card.Window.window; function onWidthChanged() { cornerMask.requestPaint() } function onHeightChanged() { cornerMask.requestPaint() } }
        }
    }
}
