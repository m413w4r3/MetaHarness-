package com.m413w4r3.metaharnessremote.data

import java.util.Base64

/**
 * Serialization of one encrypted token: the random IV and the ciphertext,
 * both base64 encoded and kept together under a version prefix.
 *
 * Pure JVM, so the format is unit tested without the Keystore, which only
 * exists on a device.
 */
internal object SecureTokenPayload {

    private const val VERSION = "v1"
    private const val SEPARATOR = "."
    private const val IV_SIZE_BYTES = 12

    fun encode(iv: ByteArray, ciphertext: ByteArray): String =
        "$VERSION$SEPARATOR${Base64.getEncoder().encodeToString(iv)}" +
            "$SEPARATOR${Base64.getEncoder().encodeToString(ciphertext)}"

    /** Returns the parts of [value], or null when it is absent or malformed. */
    fun decode(value: String?): Payload? {
        val parts = value?.split(SEPARATOR) ?: return null
        if (parts.size != 3 || parts[0] != VERSION) return null
        val iv = decodeBase64(parts[1]) ?: return null
        val ciphertext = decodeBase64(parts[2]) ?: return null
        if (iv.size != IV_SIZE_BYTES || ciphertext.isEmpty()) return null
        return Payload(iv, ciphertext)
    }

    private fun decodeBase64(value: String): ByteArray? =
        try {
            Base64.getDecoder().decode(value)
        } catch (_: IllegalArgumentException) {
            null
        }

    class Payload(val iv: ByteArray, val ciphertext: ByteArray)
}
