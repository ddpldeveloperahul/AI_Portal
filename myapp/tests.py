from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class AuthFlowTests(TestCase):
    def test_signup_creates_user_and_logs_in(self):
        response = self.client.post(
            reverse("signup"),
            {
                "username": "testuser",
                "email": "test@example.com",
                "password1": "StrongPass123!",
                "password2": "StrongPass123!",
            },
        )

        self.assertRedirects(response, reverse("base"))
        user = get_user_model().objects.get(email="test@example.com")
        self.assertEqual(user.username, "testuser")
        self.assertEqual(str(self.client.session["_auth_user_id"]), str(user.pk))

    def test_login_accepts_registered_email(self):
        get_user_model().objects.create_user(
            username="person@example.com",
            email="person@example.com",
            password="StrongPass123!",
        )

        response = self.client.post(
            reverse("login"),
            {
                "email": "person@example.com",
                "password": "StrongPass123!",
            },
        )

        self.assertRedirects(response, reverse("base"))
        self.assertIn("_auth_user_id", self.client.session)

    def test_logout_clears_session(self):
        user = get_user_model().objects.create_user(
            username="person@example.com",
            email="person@example.com",
            password="StrongPass123!",
        )
        self.client.force_login(user)

        response = self.client.get(reverse("logout"))

        self.assertRedirects(response, reverse("login"))
        self.assertNotIn("_auth_user_id", self.client.session)
