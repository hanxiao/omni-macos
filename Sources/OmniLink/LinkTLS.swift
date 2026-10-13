import Foundation
import Network
import Security

/// TLS 1.3 for the link, both ends presenting a certificate and both deciding trust by the KEY in
/// the other's certificate alone. The system's chain evaluation is never consulted: these are
/// self-signed, and the only thing that makes a peer trusted is that the user paired with that key.
enum LinkTLS {
    static let alpn = "omni-link/1"
    static let exporterLabel = "EXPORTER-omni-link-pairing"

    /// `accept` sees the peer's X9.63 public key and answers whether this handshake may complete.
    static func parameters(identity: LinkIdentity, queue: DispatchQueue,
                           accept: @escaping @Sendable (Data) -> Bool) -> NWParameters {
        let tls = NWProtocolTLS.Options()
        let so = tls.securityProtocolOptions
        sec_protocol_options_set_min_tls_protocol_version(so, .TLSv13)
        sec_protocol_options_add_tls_application_protocol(so, alpn)
        if let ident = sec_identity_create(identity.secIdentity) {
            sec_protocol_options_set_local_identity(so, ident)
        }
        // On the server this is what makes it ask for the client's certificate at all.
        sec_protocol_options_set_peer_authentication_required(so, true)
        sec_protocol_options_set_verify_block(so, { _, trust, complete in
            guard let key = peerKey(trust: sec_trust_copy_ref(trust).takeRetainedValue()) else {
                complete(false); return
            }
            complete(accept(key))
        }, queue)
        let tcp = NWProtocolTCP.Options()
        tcp.noDelay = true
        // A peer that vanishes without a FIN (lid closed, cable pulled) is noticed in ~20 s rather
        // than whenever the next request times out.
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = 10
        tcp.keepaliveInterval = 5
        tcp.keepaliveCount = 2
        tcp.connectionTimeout = 5
        let params = NWParameters(tls: tls, tcp: tcp)
        params.includePeerToPeer = false
        return params
    }

    static func peerKey(trust: SecTrust) -> Data? {
        guard let chain = SecTrustCopyCertificateChain(trust) as? [SecCertificate],
              let leaf = chain.first else { return nil }
        return key(of: leaf)
    }

    static func key(of cert: SecCertificate) -> Data? {
        guard let k = SecCertificateCopyKey(cert),
              let d = SecKeyCopyExternalRepresentation(k, nil) as Data? else { return nil }
        return d
    }

    /// The peer's key from an established connection's TLS metadata.
    static func peerKey(_ connection: NWConnection) -> Data? {
        guard let meta = connection.metadata(definition: NWProtocolTLS.definition) as? NWProtocolTLS.Metadata
        else { return nil }
        var out: Data?
        sec_protocol_metadata_access_peer_certificate_chain(meta.securityProtocolMetadata) { cert in
            if out == nil { out = key(of: sec_certificate_copy_ref(cert).takeRetainedValue()) }
        }
        return out
    }

    /// RFC 8446 exporter for this session: the same 32 bytes on both ends of ONE TLS session, and
    /// different on each leg of a relay. The pairing code covers it.
    static func exporter(_ connection: NWConnection) -> Data? {
        guard let meta = connection.metadata(definition: NWProtocolTLS.definition) as? NWProtocolTLS.Metadata
        else { return nil }
        let label = Array(exporterLabel.utf8CString)
        let secret = label.withUnsafeBufferPointer { p in
            sec_protocol_metadata_create_secret(meta.securityProtocolMetadata, exporterLabel.utf8.count, p.baseAddress!, 32)
        }
        guard let secret else { return nil }
        var d = Data()
        (secret as DispatchData).enumerateBytes { buf, _, _ in d.append(contentsOf: buf) }
        return d
    }
}
