import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatWidget from '../../components/ChatWidget';

export default function Layout(props) {
  return (
    <OriginalLayout {...props}>
      {props.children}
      {/* Floating chat widget that appears on all pages */}
      <ChatWidget />
    </OriginalLayout>
  );
}