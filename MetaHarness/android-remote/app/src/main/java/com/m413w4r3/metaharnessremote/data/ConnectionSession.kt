package com.m413w4r3.metaharnessremote.data

/**
 * Connection credentials of the current process, typed on the Settings screen
 * and read by the other screens.
 *
 * The remote token is only ever clear here: [SecureTokenStore] keeps the
 * on-disk copy sealed under an Android Keystore key. The session is primed from
 * that store at launch and updated by the Settings screen. Only the main thread
 * reads or writes it.
 */
class ConnectionSession {

    /** The remote token, or an empty string while the operator has not typed one. */
    var remoteToken: String = ""

    companion object {
        /** The single session of this process. */
        val shared = ConnectionSession()
    }
}
