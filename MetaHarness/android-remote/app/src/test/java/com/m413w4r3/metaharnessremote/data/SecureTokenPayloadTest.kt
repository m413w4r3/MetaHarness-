package com.m413w4r3.metaharnessremote.data

import java.util.Base64
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SecureTokenPayloadTest {

    private val iv = ByteArray(12) { it.toByte() }
    private val ciphertext = byteArrayOf(0, 1, -2, 127, -128)

    private fun base64(value: ByteArray): String = Base64.getEncoder().encodeToString(value)

    @Test
    fun `encode then decode round trips the iv and the ciphertext`() {
        val payload = SecureTokenPayload.decode(SecureTokenPayload.encode(iv, ciphertext))

        requireNotNull(payload)
        assertArrayEquals(iv, payload.iv)
        assertArrayEquals(ciphertext, payload.ciphertext)
    }

    @Test
    fun `encode writes a versioned payload with the two parts`() {
        val encoded = SecureTokenPayload.encode(iv, ciphertext)

        assertEquals(listOf("v1", base64(iv), base64(ciphertext)), encoded.split("."))
    }

    @Test
    fun `each encryption keeps its own iv`() {
        val otherIv = iv.copyOf().also { it[0] = 99 }

        assertNotEquals(
            SecureTokenPayload.encode(iv, ciphertext),
            SecureTokenPayload.encode(otherIv, ciphertext),
        )
    }

    @Test
    fun `decode rejects a missing payload`() {
        assertNull(SecureTokenPayload.decode(null))
        assertNull(SecureTokenPayload.decode(""))
    }

    @Test
    fun `decode rejects a foreign version`() {
        assertNull(SecureTokenPayload.decode("v2.${base64(iv)}.${base64(ciphertext)}"))
    }

    @Test
    fun `decode rejects a payload whose parts are missing`() {
        assertNull(SecureTokenPayload.decode(base64(iv)))
        assertNull(SecureTokenPayload.decode("v1.${base64(iv)}"))
        assertNull(
            SecureTokenPayload.decode("v1.${base64(iv)}.${base64(ciphertext)}.${base64(iv)}"),
        )
    }

    @Test
    fun `decode rejects parts that are not base64`() {
        assertNull(SecureTokenPayload.decode("v1.not base64.${base64(ciphertext)}"))
        assertNull(SecureTokenPayload.decode("v1.${base64(iv)}.not base64"))
    }

    @Test
    fun `decode rejects an iv that is not twelve bytes or an empty ciphertext`() {
        assertNull(SecureTokenPayload.decode("v1.${base64(iv.copyOf(11))}.${base64(ciphertext)}"))
        assertNull(SecureTokenPayload.decode("v1.${base64(iv)}."))
    }
}
