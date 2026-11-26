pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts

RowLayout {
    id: row
    property string label: ""
    property string value: ""
    property color tint: Theme.violet
    spacing: 12
    Rectangle { Layout.preferredWidth: 4; Layout.preferredHeight: 32; radius: 2; color: parent.tint; opacity: 0.85 }
    ColumnLayout { Layout.fillWidth: true; spacing: 2
        Text { text: row.label; color: Theme.muted; font.pixelSize: 12 }
        Text { text: row.value; color: Theme.text; font.pixelSize: 16; font.weight: Font.DemiBold; elide: Text.ElideRight; Layout.fillWidth: true }
    }
    Row { spacing: 3; Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
        Repeater { model: 5; delegate: Rectangle { id: level; required property int index; width: 7; height: 6 + level.index * 2; radius: 2; anchors.bottom: parent.bottom; color: row.tint; opacity: row.value === "Ожидание" || row.value === "Ожидание сигнала" ? 0.22 : 0.9 } }
    }
}
