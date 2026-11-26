import QtQuick
import QtQuick.Layouts

GlassCard {
    id: panel
    property var backend: null
    readonly property var signalLabels: backend ? backend.visibleSignals || [] : []
    readonly property var closure: backend ? backend.eyeClosurePercent : null
    readonly property var blinks: backend ? backend.blinkCount : null
    implicitHeight: signalsContent.implicitHeight + 28

    component SignalLabel: Text {
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        Layout.alignment: Qt.AlignTop
        color: Theme.muted
        font.pixelSize: 14
        textFormat: Text.PlainText
        wrapMode: Text.Wrap
    }

    component SignalValue: Text {
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        Layout.alignment: Qt.AlignTop
        color: Theme.text
        font.pixelSize: 14
        textFormat: Text.PlainText
        wrapMode: Text.Wrap
    }

    ColumnLayout {
        id: signalsContent
        anchors.left: parent.left
        anchors.right: parent.right
        spacing: 12

        SignalValue {
            text: "Глаза"
            font.pixelSize: 18
            font.weight: Font.DemiBold
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4

            SignalValue {
                text: panel.backend && panel.backend.eyeStateText || "Не уверен"
                font.pixelSize: 16
                font.weight: Font.DemiBold
            }
            SignalLabel {
                visible: text.length > 0
                text: panel.backend ? panel.backend.eyeStateReason || "" : "Камера не запущена."
            }
        }

        GridLayout {
            Layout.fillWidth: true
            Layout.maximumWidth: implicitWidth
            columns: 2
            columnSpacing: 24
            rowSpacing: 8

            SignalLabel { text: "Левый глаз" }
            SignalValue { text: panel.backend && panel.backend.leftEyeText || "Не уверен" }
            SignalLabel { text: "Правый глаз" }
            SignalValue { text: panel.backend && panel.backend.rightEyeText || "Не уверен" }
            SignalLabel { text: "Степень закрытия глаз" }
            SignalValue { text: typeof panel.closure === "number" ? panel.closure + " %" : "—" }
            SignalLabel { text: "Обнаруженных морганий с запуска камеры" }
            SignalValue { text: typeof panel.blinks === "number" ? String(panel.blinks) : "—" }
        }

        SignalLabel {
            text: "Счётчик фиксирует моргания, которые видны в нескольких последовательных кадрах."
        }

        ColumnLayout {
            Layout.fillWidth: true
            visible: panel.signalLabels.length > 0
            spacing: 8

            SignalValue {
                text: "Движения лица"
                font.pixelSize: 16
                font.weight: Font.DemiBold
            }
            Repeater {
                model: panel.signalLabels
                delegate: SignalValue {
                    required property string modelData
                    text: modelData
                }
            }
        }
    }
}
