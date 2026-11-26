import QtQuick
import QtQuick.Layouts

GlassCard {
    id: panel
    lightColor: "#886aff"
    lightStrength: 0.28
    property string observationText: ""
    RowLayout {
        anchors.fill: parent; spacing: 14
        Rectangle { Layout.preferredWidth: 48; Layout.preferredHeight: 48; radius: 14; color: "#292859"; clip: true
            Image { anchors.fill: parent; anchors.margins: 11; source: "../assets/icons/pulse.svg"; fillMode: Image.PreserveAspectFit; smooth: true }
        }
        Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: "#455174" }
        ColumnLayout { Layout.fillWidth: true; spacing: 4
            Text { text: "НАБЛЮДЕНИЕ"; color: Theme.violet; font.pixelSize: 12; font.weight: Font.DemiBold }
            Text { text: panel.observationText; color: Theme.text; font.pixelSize: 13; wrapMode: Text.WordWrap; Layout.fillWidth: true; maximumLineCount: 2; elide: Text.ElideRight }
        }
    }
}
