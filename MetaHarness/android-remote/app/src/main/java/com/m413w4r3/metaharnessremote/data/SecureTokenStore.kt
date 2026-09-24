package com.m413w4r3.metaharnessremote.data

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.GeneralSecurityException
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Encrypted storage of the remote token.
 *
 * The token is sealed with an AES/GCM key that lives in the `AndroidKeyStore`
 * and never leaves it: only the random IV and the ciphertext are written to
 * private `SharedPreferences`, so the clear token never reaches the disk.
 *
 * The key is created on the first save and loaded afterwards; every encryption
 * takes the fresh random IV that the cipher generates during `init`.
 */
class SecureTokenStore(context: Context) {

    private val preferences =
        context.applicationContext.getSharedPreferences(FILE_NAME, Context.MODE_PRIVATE)

    /** Encrypts [token] and replaces the stored payload; an empty token clears it. */
    fun saveRemoteToken(token: String) {
        if (token.isEmpty()) {
            clearRemoteToken()
            return
        }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, loadOrCreateKey())
        val ciphertext = cipher.doFinal(token.toByteArray(Charsets.UTF_8))
        preferences.edit()
            .putString(KEY_TOKEN, SecureTokenPayload.encode(cipher.iv, ciphertext))
            .apply()
    }

    /** Decrypts the stored token, or returns null when nothing readable is stored. */
    fun loadRemoteToken(): String? {
        val encoded = preferences.getString(KEY_TOKEN, null) ?: return null
        val payload = SecureTokenPayload.decode(encoded)
        val key = payload?.let { loadKey() }
        if (payload == null || key == null) {
            // A payload that this device cannot read is useless: drop it.
            clearRemoteToken()
            return null
        }
        return try {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(TAG_LENGTH_BITS, payload.iv))
            String(cipher.doFinal(payload.ciphertext), Charsets.UTF_8)
        } catch (_: GeneralSecurityException) {
            clearRemoteToken()
            null
        }
    }

    /** Forgets the stored payload; the Keystore key stays for the next save. */
    fun clearRemoteToken() {
        preferences.edit().remove(KEY_TOKEN).apply()
    }

    private fun loadOrCreateKey(): SecretKey = loadKey() ?: generateKey()

    private fun loadKey(): SecretKey? {
        val keyStore = KeyStore.getInstance(PROVIDER).apply { load(null) }
        return keyStore.getKey(KEY_ALIAS, null) as? SecretKey
    }

    private fun generateKey(): SecretKey {
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, PROVIDER)
        generator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(KEY_SIZE_BITS)
                .build(),
        )
        return generator.generateKey()
    }

    private companion object {
        const val PROVIDER = "AndroidKeyStore"
        const val KEY_ALIAS = "metaharness_remote_token"
        const val TRANSFORMATION = "AES/GCM/NoPadding"
        const val KEY_SIZE_BITS = 256
        const val TAG_LENGTH_BITS = 128
        const val FILE_NAME = "metaharness_remote_credentials"
        const val KEY_TOKEN = "remote_token"
    }
}
