package com.m413w4r3.metaharnessremote.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GatewayUrlsTest {

    @Test
    fun `a bare tailnet host is normalised to https`() {
        assertEquals(
            GatewayUrls.BaseUrl.Valid("https://my-pc.my-tailnet.ts.net"),
            GatewayUrls.normalize("  my-pc.my-tailnet.ts.net  "),
        )
    }

    @Test
    fun `port is kept and the trailing slash is dropped`() {
        assertEquals(
            GatewayUrls.BaseUrl.Valid("https://my-pc.my-tailnet.ts.net:8443"),
            GatewayUrls.normalize("https://my-pc.my-tailnet.ts.net:8443/"),
        )
    }

    @Test
    fun `cleartext http is refused`() {
        val result = GatewayUrls.normalize("http://my-pc.my-tailnet.ts.net")

        assertEquals(GatewayUrls.BaseUrl.Invalid("Server URL must use https"), result)
    }

    @Test
    fun `a blank url is refused`() {
        assertTrue(GatewayUrls.normalize("   ") is GatewayUrls.BaseUrl.Invalid)
    }

    @Test
    fun `a url that embeds credentials is refused`() {
        val result = GatewayUrls.normalize("https://user:token@my-pc.my-tailnet.ts.net")

        assertTrue(result is GatewayUrls.BaseUrl.Invalid)
    }

    @Test
    fun `the health route is appended to the normalised base url`() {
        assertEquals(
            "https://my-pc.my-tailnet.ts.net/v1/health",
            GatewayUrls.healthUrl("https://my-pc.my-tailnet.ts.net/"),
        )
    }
}
