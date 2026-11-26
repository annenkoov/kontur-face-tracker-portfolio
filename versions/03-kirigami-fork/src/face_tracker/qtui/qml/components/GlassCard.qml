import QtQuick

Rectangle {
    id: card
    default property alias content: contentHost.data
    property color tint: Theme.panel
    property color lightColor: "#8175ff"
    property real lightStrength: 0.08
    property int contentMargin: 18
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
        // Полноразмерная подложка: градиент проходит по всей карточке,
        // а не сводится к узкой полосе у её края.
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: Qt.rgba(card.lightColor.r, card.lightColor.g, card.lightColor.b, card.lightStrength * 0.04) }
            GradientStop { position: 0.52; color: Qt.rgba(card.lightColor.r, card.lightColor.g, card.lightColor.b, card.lightStrength * 0.22) }
            GradientStop { position: 1.0; color: Qt.rgba(card.lightColor.r, card.lightColor.g, card.lightColor.b, card.lightStrength * 0.56) }
        }
        opacity: 1.0
        border.width: 0
        clip: true
    }

    Rectangle {
        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
        anchors.leftMargin: card.radius; anchors.rightMargin: card.radius
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
        anchors.margins: card.contentMargin
        clip: true
    }
}
