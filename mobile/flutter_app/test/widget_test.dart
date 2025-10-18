// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:potato_flutter_app/main.dart';

void main() {
  testWidgets('App shows title and placeholder', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const PotatoApp());

    // Expect the app bar title to be present
    expect(find.text('Potato Leaf Detector'), findsOneWidget);

    // Expect the initial placeholder text for image preview
    expect(find.text('No image selected'), findsOneWidget);
  });
}
