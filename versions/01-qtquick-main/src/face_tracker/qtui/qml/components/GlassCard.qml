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
            GradientStop { position: 0.0; color: Qt.rgba(0.49, 0.43, 0.98, 0.015) }
            GradientStop { position: 0.65; color: Qt.rgba(0.49, 0.43, 0.98, 0.035) }
            GradientStop { position: 0.90; color: Qt.rgba(0.49, 0.43, 0.98, 0.10) }
            GradientStop { position: 1.0; color: Qt.rgba(0.26, 0.82, 0.78, 0.13) }
        }
        opacity: 1.0
        border.width: 0
        clip: true
    }

    Rectangle {
        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
        height: 1; radius: 1
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0; color: "#12ffffff" }
            GradientStop { position: 0.45; color: "#24c3d3ff" }
            GradientStop { position: 1; color: "#12ffffff" }
        }
    }

    Item {
        id: contentHost
        anchors.fill: parent
        anchors.margins: 18
        clip: true
    }
}
