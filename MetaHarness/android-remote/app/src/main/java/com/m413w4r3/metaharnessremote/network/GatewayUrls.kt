package com.m413w4r3.metaharnessremote.network

import okhttp3.HttpUrl.Companion.toHttpUrlOrNull

/**
 * Normalises the server URL typed in Settings and derives the gateway routes.
 *
 * The gateway is only reachable through Tailscale Serve, so an absolute
 * `https` base URL is required here as well as in the manifest.
 */
object GatewayUrls {

    const val HEALTH_PATH = "/v1/health"

    sealed interface BaseUrl {
        data class Valid(val value: String) : BaseUrl
        data class Invalid(val reason: String) : BaseUrl
    }

    /** Accepts `host`, `host:port` or a full URL; a missing scheme means https. */
    fun normalize(raw: String): BaseUrl {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return BaseUrl.Invalid("Server URL is required")

        val candidate = if (trimmed.contains("://")) trimmed else "https://$trimmed"
        val parsed = candidate.toHttpUrlOrNull()
            ?: return BaseUrl.Invalid("Server URL is not a valid URL")
        if (!parsed.isHttps) return BaseUrl.Invalid("Server URL must use https")
        if (parsed.username.isNotEmpty() || parsed.password.isNotEmpty()) {
            return BaseUrl.Invalid("Server URL must not embed credentials")
        }

        val base = parsed.newBuilder()
            .query(null)
            .fragment(null)
            .build()
            .toString()
            .trimEnd('/')
        return BaseUrl.Valid(base)
    }

    /** `<baseUrl>` + [HEALTH_PATH]; the base URL is expected to be normalised. */
    fun healthUrl(baseUrl: String): String = baseUrl.trimEnd('/') + HEALTH_PATH
}
