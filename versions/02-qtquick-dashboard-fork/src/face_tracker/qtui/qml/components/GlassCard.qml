import QtQuick

Rectangle {
    id: card
    default property alias content: contentHost.data
    property color tint: Theme.panel
    radius: 18
    // Полупрозрачный тёмный материал пропускает мягкий фон, сохраняя
    // достаточный контраст для текста и не создавая тяжёлых постэффектов.
    color: Qt.rgba(tint.r, tint.g, tint.b, 0.90)
    border.width: 1
    border.color: "#46567e"
    clip: true

    Rectangle {
        id: glassTint
        anchors.fill: parent
        radius: card.radius
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: Qt.rgba(0.49, 0.43, 0.98, 0.13) }
            GradientStop { position: 0.65; color: Qt.rgba(0.49, 0.43, 0.98, 0.10) }
            GradientStop { position: 0.90; color: Qt.rgba(0.49, 0.43, 0.98, 0.15) }
            GradientStop { position: 1.0; color: Qt.rgba(0.26, 0.82, 0.78, 0.17) }
        }
        opacity: 1.0
        border.width: 0
        clip: true
    }

    Item {
        id: contentHost
        anchors.fill: parent
        anchors.margins: 14
        clip: true
    }
}
