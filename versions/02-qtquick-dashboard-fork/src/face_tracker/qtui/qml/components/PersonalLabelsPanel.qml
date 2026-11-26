import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: panel
    property var backend: null
    readonly property var frame: backend && backend.sampleFrame || ({})
    readonly property var summary: backend && backend.personalSummary || ({})
    readonly property var labels: backend && backend.expressionLabels || []
    readonly property var samples: backend && backend.expressionSamples || []
    readonly property var feedback: backend && backend.sampleFeedback || ({})

    spacing: 16

    component Copy: Text {
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        color: Theme.muted
        font.pixelSize: 14
        wrapMode: Text.Wrap
        textFormat: Text.PlainText
    }

    component Title: Copy {
        color: Theme.text
        font.pixelSize: 18
        font.weight: Font.DemiBold
    }

    component LabelButton: Button {
        id: control
        implicitHeight: 40
        leftPadding: 14
        rightPadding: 14
        focusPolicy: Qt.StrongFocus
        background: Rectangle {
            radius: 12
            color: control.hovered || control.down ? Theme.border : Theme.panelRaised
            border.color: control.activeFocus ? Theme.text : Theme.border
            border.width: control.activeFocus ? 2 : 1
        }
        contentItem: Text {
            text: control.text
            color: control.enabled ? Theme.text : Theme.muted
            font.pixelSize: 14
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        opacity: enabled ? 1 : 0.7
    }

    component LabelDialog: Dialog {
        id: dialog
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(540, parent ? parent.width - 48 : 540)
        padding: 20
        modal: true
        focus: true
        background: Rectangle { radius: 18; color: Theme.panel; border.color: Theme.border }
        header: Text {
            padding: 20
            text: dialog.title
            color: Theme.text
            font.pixelSize: 18
            font.weight: Font.DemiBold
        }
    }

    GlassCard {
        Layout.fillWidth: true
        implicitHeight: personalContent.implicitHeight + 28

        ColumnLayout {
            id: personalContent
            anchors.left: parent.left
            anchors.right: parent.right
            spacing: 12

            Title { text: "Личная метка" }
            Copy { text: "Отметь, как ты себя чувствуешь. Приложение сохранит обрезку лица и данные о мимике на этом компьютере." }
            Copy {
                color: Theme.text
                text: panel.summary.status || "Недостаточно примеров"
                font.weight: Font.DemiBold
            }
            Copy {
                text: panel.frame.prediction ? "Подсказка по твоим отметкам · " + panel.frame.prediction : ""
                visible: text.length > 0
                color: Theme.text
            }
            Copy {
                text: panel.summary.reason || (panel.frame.prediction ? panel.frame.predictionReason : "") || ""
                visible: text.length > 0
            }
            Copy {
                text: panel.frame.personalIssue && panel.frame.personalIssue !== panel.frame.frameText
                    ? panel.frame.frameText + " · " + panel.frame.personalIssue
                    : panel.frame.frameText || "Начни сессию, чтобы сохранить кадр."
            }

            RowLayout {
                spacing: 8
                Repeater {
                    model: panel.summary.labels || []
                    delegate: LabelButton {
                        required property var modelData
                        text: "Сохранить «" + modelData.label + "»"
                        enabled: panel.backend && panel.frame.personalIssue === ""
                        onClicked: panel.backend.savePersonalSample(modelData.label)
                    }
                }
            }

            Copy {
                text: "Учтено отдельных сессий · " + (panel.summary.sessions || 0)
                color: Theme.text
            }
            Repeater {
                model: panel.summary.labels || []
                delegate: Copy {
                    required property var modelData
                    text: modelData.label + " · кадров " + modelData.count
                        + " · сессий для обучения " + modelData.train + ", для проверки " + modelData.validation
                }
            }
            Copy { text: panel.summary.requirements || "" }
            Copy {
                text: panel.feedback.personal || ""
                visible: text.length > 0
                color: Theme.text
            }
            LabelButton {
                text: "Удалить личные примеры"
                enabled: panel.summary.storeAvailable === true
                onClicked: clearDialog.open()
            }
        }
    }

    GlassCard {
        Layout.fillWidth: true
        implicitHeight: expressionContent.implicitHeight + 28

        ColumnLayout {
            id: expressionContent
            anchors.left: parent.left
            anchors.right: parent.right
            spacing: 12

            Title { text: "Примеры выражений лица" }
            Copy { text: "Отметь видимое выражение для сравнения моделей. Метка описывает мимику, без оценки стресса или диагноза." }
            Copy {
                text: panel.frame.expressionIssue || panel.frame.frameText || "Начни сессию, чтобы сохранить кадр."
            }
            LabelButton {
                text: "Проверить кадр"
                enabled: panel.backend && panel.frame.expressionIssue === ""
                onClicked: if (panel.backend.prepareExpressionSample()) captureDialog.open()
            }
            Copy {
                text: panel.feedback.expression || ""
                visible: text.length > 0
                color: Theme.text
            }
            LabelButton {
                text: "Просмотреть примеры · " + panel.samples.length
                enabled: panel.samples.length > 0
                onClicked: reviewDialog.open()
            }
        }
    }

    LabelDialog {
        id: clearDialog
        title: "Удалить личные примеры?"
        contentItem: ColumnLayout {
            spacing: 16
            Copy { text: "Обрезки лица и данные о мимике для личных меток будут удалены. Действие нельзя отменить." }
            RowLayout {
                Layout.alignment: Qt.AlignRight
                spacing: 8
                LabelButton { text: "Отмена"; onClicked: clearDialog.close() }
                LabelButton {
                    text: "Удалить личные примеры"
                    onClicked: {
                        clearDialog.close()
                        if (panel.backend)
                            panel.backend.clearPersonalSamples()
                    }
                }
            }
        }
    }

    LabelDialog {
        id: captureDialog
        title: "Какое выражение видно?"
        onOpened: newExpression.currentIndex = -1
        onClosed: if (panel.backend) panel.backend.discardExpressionDraft()

        contentItem: ColumnLayout {
            spacing: 12
            Copy { text: "Проверь обрезку. Она сохранится на этом компьютере после подтверждения." }
            Image {
                id: draftPreview
                Layout.fillWidth: true
                Layout.preferredHeight: 220
                visible: status === Image.Ready
                cache: false
                fillMode: Image.PreserveAspectFit
                source: captureDialog.visible && panel.backend ? panel.backend.expressionDraftUrl : ""
                Accessible.name: "Обрезка лица для проверки"
            }
            Copy {
                visible: draftPreview.status === Image.Error
                text: "Не удалось показать обрезку. Закрой окно и попробуй снова."
            }
            ComboBox {
                id: newExpression
                Layout.preferredWidth: 220
                enabled: draftPreview.status === Image.Ready
                model: panel.labels
                textRole: "title"
                valueRole: "key"
                displayText: currentIndex < 0 ? "Выбери выражение" : currentText
                Accessible.name: "Выражение на обрезке"
            }
            Copy {
                text: panel.feedback.expression || ""
                visible: text.length > 0
                color: Theme.text
            }
            RowLayout {
                Layout.alignment: Qt.AlignRight
                spacing: 8
                LabelButton { text: "Отмена"; onClicked: captureDialog.close() }
                LabelButton {
                    text: "Сохранить пример"
                    enabled: draftPreview.status === Image.Ready && newExpression.currentIndex >= 0
                    onClicked: {
                        if (panel.backend.confirmExpressionSample(newExpression.currentValue))
                            captureDialog.close()
                    }
                }
            }
        }
    }

    LabelDialog {
        id: reviewDialog
        title: "Сохранённые выражения"
        property var selected: panel.samples[samplePicker.currentIndex] || null
        property string pendingDelete: ""
        onSelectedChanged: {
            pendingDelete = ""
            if (correction)
                correction.currentIndex = selected ? correction.indexOfValue(selected.label) : -1
        }
        onOpened: {
            samplePicker.currentIndex = panel.samples.length ? 0 : -1
            correction.currentIndex = selected ? correction.indexOfValue(selected.label) : -1
        }
        onClosed: pendingDelete = ""

        contentItem: ColumnLayout {
            spacing: 12
            ComboBox {
                id: samplePicker
                Layout.fillWidth: true
                model: panel.samples
                textRole: "description"
                Accessible.name: "Сохранённый кадр"
            }
            Image {
                id: preview
                visible: status === Image.Ready
                Layout.fillWidth: true
                Layout.preferredHeight: 180
                cache: false
                asynchronous: true
                fillMode: Image.PreserveAspectFit
                source: reviewDialog.visible && panel.backend && reviewDialog.selected
                    ? panel.backend.expressionPreview(reviewDialog.selected.id) : ""
                Accessible.name: "Сохранённая обрезка лица"
            }
            Copy {
                visible: preview.status === Image.Null || preview.status === Image.Error
                text: "Обрезка лица недоступна."
            }
            RowLayout {
                visible: reviewDialog.pendingDelete === ""
                spacing: 8
                ComboBox {
                    id: correction
                    Layout.preferredWidth: 200
                    model: panel.labels
                    textRole: "title"
                    valueRole: "key"
                    Accessible.name: "Исправленная метка выражения"
                }
                LabelButton {
                    text: "Исправить метку"
                    enabled: reviewDialog.selected && correction.currentIndex >= 0
                        && correction.currentValue !== reviewDialog.selected.label
                    onClicked: {
                        panel.backend.updateExpressionSample(reviewDialog.selected.id, correction.currentValue)
                        reviewDialog.close()
                    }
                }
            }
            Copy {
                visible: reviewDialog.pendingDelete !== ""
                text: "Удалить выбранный кадр и его метку? Действие нельзя отменить."
            }
            RowLayout {
                Layout.alignment: Qt.AlignRight
                spacing: 8
                LabelButton { text: "Закрыть"; onClicked: reviewDialog.close() }
                LabelButton {
                    text: reviewDialog.pendingDelete ? "Подтвердить удаление" : "Удалить пример"
                    enabled: reviewDialog.selected !== null
                    onClicked: {
                        if (!reviewDialog.pendingDelete) {
                            reviewDialog.pendingDelete = reviewDialog.selected.id
                            return
                        }
                        panel.backend.deleteExpressionSample(reviewDialog.pendingDelete)
                        reviewDialog.close()
                    }
                }
            }
        }
    }
}
