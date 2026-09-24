package com.m413w4r3.metaharnessremote.data

/**
 * Connection credentials of the current process, typed on the Settings screen
 * and read by the other screens.
 *
 * The remote token lives here and nowhere else: it is never written to disk,
 * so restarting the app starts from an empty token again. Only the main thread
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
