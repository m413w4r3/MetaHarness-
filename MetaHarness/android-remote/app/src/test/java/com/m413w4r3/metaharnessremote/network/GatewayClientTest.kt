package com.m413w4r3.metaharnessremote.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GatewayClientTest {

    private val baseUrl = "https://my-pc.my-tailnet.ts.net"
    private val token = "remote-token-value"

    @Test
    fun `health request is a GET on the health route`() {
        val request = GatewayClient.healthRequest(baseUrl, token)

        assertEquals("GET", request.method)
        assertEquals("$baseUrl/v1/health", request.url.toString())
    }

    @Test
    fun `health request carries the bearer token in the authorization header`() {
        val request = GatewayClient.healthRequest(baseUrl, token)

        assertEquals("Bearer $token", request.header("Authorization"))
    }

    @Test
    fun `the token never leaks into the url`() {
        val request = GatewayClient.healthRequest(baseUrl, token)

        assertFalse(request.url.toString().contains(token))
    }

    @Test
    fun `http client is offline-safe, never retries and logs no header`() {
        val client = GatewayClient.defaultHttpClient()

        assertFalse("mutations must never be retried", client.retryOnConnectionFailure)
        assertFalse("the token must not follow a redirect", client.followRedirects)
        assertFalse("the token must not follow an ssl redirect", client.followSslRedirects)
        assertTrue("no logging interceptor", client.interceptors.isEmpty())
        assertTrue("no logging network interceptor", client.networkInterceptors.isEmpty())
        assertEquals(10_000, client.connectTimeoutMillis)
        assertEquals(20_000, client.readTimeoutMillis)
        assertEquals(30_000, client.callTimeoutMillis)
    }
}
