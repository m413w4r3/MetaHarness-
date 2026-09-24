package com.m413w4r3.metaharnessremote.ui.rundetail

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

/**
 * Placeholder of the Run Detail screen: it shows which run was opened.
 *
 * A later prompt replaces this with the detail view itself; the route and the
 * run-id hand-off are already the ones it will use, so a run created on the
 * phone lands here.
 */
@Composable
fun RunDetailScreen(runId: String, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Run Detail",
            style = MaterialTheme.typography.headlineSmall,
        )
        Text(
            text = runId,
            style = MaterialTheme.typography.titleMedium,
            fontFamily = FontFamily.Monospace,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        Text(
            text = "The run detail view arrives with a later prompt.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
