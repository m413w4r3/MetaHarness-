package com.m413w4r3.metaharnessremote.data

import android.content.Context

/**
 * Persists the non-secret part of the connection settings.
 *
 * Only the server URL is stored: the remote token stays in memory and is never
 * written to disk.
 */
class ServerSettingsStore(context: Context) {

    private val preferences =
        context.applicationContext.getSharedPreferences(FILE_NAME, Context.MODE_PRIVATE)

    fun loadServerUrl(): String = preferences.getString(KEY_SERVER_URL, "").orEmpty()

    fun saveServerUrl(serverUrl: String) {
        preferences.edit().putString(KEY_SERVER_URL, serverUrl).apply()
    }

    private companion object {
        const val FILE_NAME = "metaharness_remote_settings"
        const val KEY_SERVER_URL = "server_url"
    }
}
